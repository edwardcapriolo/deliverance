package io.teknek.deliverance.model.nemotronlabsdiffusion;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.CausalSelfAttention;
import io.teknek.deliverance.generator.EmbedInput;
import io.teknek.deliverance.generator.FinishReason;
import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.KvCacheSelfAttention;
import io.teknek.deliverance.generator.LayerNorm;
import io.teknek.deliverance.generator.MLPBlock;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.generator.RmsNorm;
import io.teknek.deliverance.generator.SampleOutput;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.grace.TokenIds;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.model.TensorProviderKind;
import io.teknek.deliverance.model.llama.LlamaModel;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.safetensors.fetch.LoraAdapterModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.kv.CacheExecutionMode;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.UUID;
import java.util.stream.IntStream;

import static io.teknek.deliverance.tensor.AbstractTensorUtils.quantize;

/**
 * NVIDIA Nemotron-Labs-Diffusion model support.
 *
 * <p>The first executable path is upstream's AR baseline: it uses the shared `encoder.*` weights, causal attention, KV
 * cache, final `encoder.norm`, and `diffusion_head.weight` logits. Diffusion decoding is the project goal, but AR is the
 * required same-model baseline for CPU throughput comparison.</p>
 */
public class NemotronLabsDiffusionModel extends LlamaModel {
    private static final Logger LOGGER = LoggerFactory.getLogger(NemotronLabsDiffusionModel.class);
    private static final String LINEAR_SPEC_LORA_ADAPTER_ID = "nemotron_linear_spec_lora";

    private volatile AbstractTensor embedTokenWeights;
    private AbstractTensor[] inputNormWeights;
    private AbstractTensor[] queryWeights;
    private AbstractTensor[] keyWeights;
    private AbstractTensor[] valueWeights;
    private AbstractTensor[] outputWeights;
    private AbstractTensor[] postAttentionNormWeights;
    private AbstractTensor[] gateWeights;
    private AbstractTensor[] upWeights;
    private AbstractTensor[] downWeights;
    private KvCacheSelfAttention[] kvCacheSelfAttentions;
    private MLPBlock[] mlpBlocks;
    private AbstractTensor finalNormWeight;
    private AbstractTensor diffusionHeadWeight;
    private final NemotronLabsDiffusionRope rope;
    private boolean linearSpecLoraAdapterRegistered;

    public NemotronLabsDiffusionModel(InferenceType inferenceType, Config config, WeightLoader weights,
            PreTrainedTokenizer tokenizer, DType workingMemoryDType, DType workingMemoryQType,
            Optional<DType> modelQType, ConfigurableTensorProvider configurableTensorProvider,
            MetricRegistry metricRegistry, TensorAllocator tensorAllocator, KvBufferCacheSettings kvBufferCacheSettings,
            ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization) {
        super(inferenceType, config, weights, tokenizer, workingMemoryDType, workingMemoryQType, modelQType,
                configurableTensorProvider, metricRegistry, tensorAllocator, kvBufferCacheSettings, toolCallParser, pool,
                tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
        this.rope = new NemotronLabsDiffusionRope((NemotronLabsDiffusionConfig) config);
    }

    /**
     * Registers NVIDIA's linear-speculation LoRA adapter for the diffusion draft phase.
     *
     * <p>The adapter is stored under {@code nvidia/Nemotron-Labs-Diffusion-3B/linear_spec_lora} upstream, even when the
     * base model under test is {@code Nemotron-Labs-Diffusion-3B-Base}. It is enabled only for bidirectional draft
     * forwards and disabled for causal prefill/verify, matching the reference implementation.</p>
     */
    public void registerLinearSpecLoraAdapter() {
        registerLoraAdapter(LINEAR_SPEC_LORA_ADAPTER_ID,
                new LoraAdapterModelFetcher("nvidia", "Nemotron-Labs-Diffusion-3B", "linear_spec_lora", true));
        linearSpecLoraAdapterRegistered = true;
    }

    @Override
    protected boolean addBosToken() {
        return false;
    }

    @Override
    protected EmbedInput loadInputWeights() {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "nemotron_labs_diffusion.embedding.load").time()) {
            if (embedTokenWeights == null) {
                String name = "encoder.embed_tokens.weight";
                LOGGER.debug("loading Nemotron embeddings weight={} target_dtype={}", name, workingDType);
                embedTokenWeights = quantize(weights.load(name), workingDType);
                registerModelLineageTensor(name, embedTokenWeights);
                configurableTensorProvider.get().registerModelTensor(embedTokenWeights);
            }
        }
        return new EmbedInput(this) {
            @Override
            public AbstractTensor inputTokenToEmbedding(int inputToken, int unused) {
                Preconditions.checkArgument(inputToken >= 0 && inputToken < embedTokenWeights.shape().first(),
                        "input token out of bounds");
                AbstractTensor row = embedTokenWeights.slice(true, inputToken);
                AbstractTensor embedding = parent.getTensorAllocator().getDirty(row.dType(), row.shape());
                embedding.copyFrom(row, 0, 0, config.embeddingLength);
                return embedding;
            }
        };
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        TransformerBlock[] blocks = new TransformerBlock[config.numberOfLayers];
        inputNormWeights = new AbstractTensor[config.numberOfLayers];
        queryWeights = new AbstractTensor[config.numberOfLayers];
        keyWeights = new AbstractTensor[config.numberOfLayers];
        valueWeights = new AbstractTensor[config.numberOfLayers];
        outputWeights = new AbstractTensor[config.numberOfLayers];
        postAttentionNormWeights = new AbstractTensor[config.numberOfLayers];
        gateWeights = new AbstractTensor[config.numberOfLayers];
        upWeights = new AbstractTensor[config.numberOfLayers];
        downWeights = new AbstractTensor[config.numberOfLayers];
        kvCacheSelfAttentions = new KvCacheSelfAttention[config.numberOfLayers];
        mlpBlocks = new MLPBlock[config.numberOfLayers];
        IntStream.range(0, config.numberOfLayers).parallel().forEach(layer -> {
            try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                    "nemotron_labs_diffusion.layer.load").time()) {
                String base = "encoder.layers." + layer + ".";
                String attentionPrefix = base + "self_attn.";
                String qName = attentionPrefix + "q_proj.weight";
                String kName = attentionPrefix + "k_proj.weight";
                String vName = attentionPrefix + "v_proj.weight";
                String oName = attentionPrefix + "o_proj.weight";
                AbstractTensor qWeight = quantize(weights.load(qName), qType);
                AbstractTensor kWeight = quantize(weights.load(kName), qType);
                AbstractTensor vWeight = quantize(weights.load(vName), qType);
                AbstractTensor oWeight = quantize(weights.load(oName), qType);
                queryWeights[layer] = qWeight;
                keyWeights[layer] = kWeight;
                valueWeights[layer] = vWeight;
                outputWeights[layer] = oWeight;
                registerModelLineageTensor(qName, qWeight);
                registerModelLineageTensor(kName, kWeight);
                registerModelLineageTensor(vName, vWeight);
                registerModelLineageTensor(oName, oWeight);

                CausalSelfAttention attention = new CausalSelfAttention(this, layer, qWeight, kWeight, vWeight,
                        oWeight, configurableTensorProvider, metricRegistry, qName, kName, vName, oName);
                kvCacheSelfAttentions[layer] = new KvCacheSelfAttention(this, layer, qWeight, kWeight, vWeight,
                        oWeight, configurableTensorProvider, metricRegistry, qName, kName, vName, oName);

                String mlpPrefix = base + "mlp.";
                String gateName = mlpPrefix + "gate_proj.weight";
                String downName = mlpPrefix + "down_proj.weight";
                String upName = mlpPrefix + "up_proj.weight";
                AbstractTensor gateWeight = quantize(weights.load(gateName), qType);
                AbstractTensor downWeight = quantize(weights.load(downName), qType);
                AbstractTensor upWeight = quantize(weights.load(upName), qType);
                gateWeights[layer] = gateWeight;
                downWeights[layer] = downWeight;
                upWeights[layer] = upWeight;
                registerModelLineageTensor(gateName, gateWeight);
                registerModelLineageTensor(downName, downWeight);
                registerModelLineageTensor(upName, upWeight);

                MLPBlock mlp = new MLPBlock(this, config.activationFunction, gateWeight, downWeight, upWeight,
                        configurableTensorProvider, gateName, upName, downName);
                mlpBlocks[layer] = mlp;

                String inputNormName = base + "input_layernorm.weight";
                String postAttentionNormName = base + "post_attention_layernorm.weight";
                AbstractTensor inputNormWeight = quantize(weights.load(inputNormName), qType);
                AbstractTensor postAttentionNormWeight = quantize(weights.load(postAttentionNormName), qType);
                inputNormWeights[layer] = inputNormWeight;
                postAttentionNormWeights[layer] = postAttentionNormWeight;
                registerModelLineageTensor(inputNormName, inputNormWeight);
                registerModelLineageTensor(postAttentionNormName, postAttentionNormWeight);

                blocks[layer] = new TransformerBlock(this, layer, new RmsNorm(this, inputNormWeight, metricRegistry),
                        attention, new RmsNorm(this, postAttentionNormWeight, metricRegistry), mlp,
                        configurableTensorProvider);
            }
        });
        return blocks;
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "nemotron_labs_diffusion.output.load").time()) {
            String normName = "encoder.norm.weight";
            AbstractTensor outputNormWeight = quantize(weights.load(normName), qType);
            finalNormWeight = outputNormWeight;
            registerModelLineageTensor(normName, outputNormWeight);
            LayerNorm outputLayerNorm = new RmsNorm(this, outputNormWeight, metricRegistry);

            String headName = "diffusion_head.weight";
            DType outputHeadDType = outputHeadQuantization.orElse(workingDType);
            AbstractTensor logitsWeight = io.teknek.deliverance.tensor.AbstractTensorUtils.quantize(
                    weights.load(headName), outputHeadDType, outputHeadQuantization.isPresent());
            diffusionHeadWeight = logitsWeight;
            registerModelLineageTensor(headName, logitsWeight);
            configurableTensorProvider.get().registerModelTensor(logitsWeight);
            return new SampleOutput() {
                @Override
                public LayerNorm getOutputLayerNorm() {
                    return outputLayerNorm;
                }

                @Override
                public AbstractTensor getOutputLogitsWeights() {
                    return logitsWeight;
                }
            };
        }
    }

    /** Runs the same-model AR baseline through the Nemotron YaRN causal path. */
    public Response generateArBaseline(UUID sessionId, PromptContext promptContext, GeneratorParameters parameters,
            GenerateEvent eventFired) {
        long startNanos = System.nanoTime();
        int maxNewTokens = Math.max(1, parameters.maxTokens.or(() -> parameters.ntokens).orElse(1));
        int[] promptTokens = constructPromptTokensForRuntime(promptContext.getPrompt());
        List<Integer> generated = new ArrayList<>(maxNewTokens);
        long firstTokenNanos = 0L;

        try (AbstractModel.TensorPlanTraceScope ignoredTrace = openTensorPlanTrace(sessionId);
             KvCacheSession kvSession = newKvCacheSession();
             AbstractTensor argMax = tensorAllocator.getDirty(DType.F32, TensorShape.of(1, 2))) {
            try (AbstractTensor promptHidden = forwardCausalWithCache(promptTokens, 0, kvSession);
                 AbstractTensor logits = logitsForHiddenRow(promptHidden, promptTokens.length - 1)) {
                configurableTensorProvider.get().argMax(logits, argMax, 0, config.vocabularySize);
                int token = (int) argMax.get(0, 0);
                generated.add(token);
                firstTokenNanos = System.nanoTime() - startNanos;
                String raw = decodeTokens(new int[] { token }, false);
                eventFired.emit(token, raw, raw, elapsedMs(startNanos));
            }
            for (int step = 0; step < maxNewTokens; step++) {
                if (generated.size() >= maxNewTokens || config.eosTokens.contains(generated.getLast())) {
                    break;
                }
                int previousToken = generated.getLast();
                int position = kvSession.length();
                try (AbstractTensor hidden = forwardCausalWithCache(new int[] { previousToken }, position, kvSession);
                     AbstractTensor logits = logitsForHiddenRow(hidden, 0)) {
                    configurableTensorProvider.get().argMax(logits, argMax, 0, config.vocabularySize);
                    int token = (int) argMax.get(0, 0);
                    generated.add(token);
                    String raw = decodeTokens(new int[] { token }, false);
                    eventFired.emit(token, raw, raw, elapsedMs(startNanos));
                    if (config.eosTokens.contains(token)) {
                        break;
                    }
                }
            }
        }

        int[] generatedIds = generated.stream().mapToInt(Integer::intValue).toArray();
        long totalMs = Math.round(elapsedMs(startNanos));
        double tokensPerSecond = totalMs <= 0 ? 0.0 : generated.size() * 1000.0 / totalMs;
        LOGGER.info("nemotron_ar_baseline_complete prompt_tokens={} generated_tokens={} total_ms={} tokens_per_second={}",
                promptTokens.length, generated.size(), totalMs, tokensPerSecond);
        FinishReason reason = !generated.isEmpty() && config.eosTokens.contains(generated.getLast())
                ? FinishReason.STOP_TOKEN
                : FinishReason.MAX_TOKENS;
        return timedResponse(new Response(decodeTokens(generatedIds, true), decodeTokens(generatedIds, false),
                reason, promptTokens.length, generated, 0, totalMs, List.of()), startNanos, firstTokenNanos);
    }

    /** Defaults Nemotron generation to diffusion mode; use {@link #generateArBaseline} for AR comparison. */
    @Override
    public boolean usesModelSpecificGeneration() {
        return generationOptions().containsKey("mode");
    }

    @Override
    public Response generate(UUID sessionId, PromptContext promptContext, GeneratorParameters parameters,
            GenerateEvent eventFired) {
        String mode = generationOptionString("mode", "linear_spec");
        if ("ar".equalsIgnoreCase(mode) || "autoregressive".equalsIgnoreCase(mode)) {
            return generateArBaseline(sessionId, promptContext, parameters, eventFired);
        }
        Preconditions.checkArgument("diffusion".equalsIgnoreCase(mode) || "linear_spec".equalsIgnoreCase(mode),
                "unsupported Nemotron generationOptions.mode: %s", mode);
        int requestedTokens = parameters.maxTokens.or(() -> parameters.ntokens).orElse(((NemotronLabsDiffusionConfig) config).blockSize);
        int configuredBlockLength = parameters.diffusionBlockLength.orElseGet(() ->
                generationOptionInt("blockLength", ((NemotronLabsDiffusionConfig) config).blockSize));
        int blockLength = Math.max(1, Math.min(configuredBlockLength, requestedTokens));
        float threshold = generationOptionFloat("threshold", 0.0f);
        return generateDiffusion(sessionId, promptContext, parameters, eventFired, blockLength, threshold);
    }

    private String generationOptionString(String key, String defaultValue) {
        Object value = generationOptions().get(key);
        return value == null ? defaultValue : value.toString();
    }

    private int generationOptionInt(String key, int defaultValue) {
        Object value = generationOptions().get(key);
        if (value == null) {
            return defaultValue;
        }
        if (value instanceof Number number) {
            return number.intValue();
        }
        return Integer.parseInt(value.toString());
    }

    private float generationOptionFloat(String key, float defaultValue) {
        Object value = generationOptions().get(key);
        if (value == null) {
            return defaultValue;
        }
        if (value instanceof Number number) {
            return number.floatValue();
        }
        return Float.parseFloat(value.toString());
    }

    /**
     * Runs upstream-style bidirectional block denoising for the Base model.
     *
     * <p>This is intentionally separate from {@link #generate(UUID, PromptContext, GeneratorParameters, GenerateEvent)},
     * which remains the AR baseline. The implementation prioritizes a measurable CPU diffusion path over AR reuse: each
     * denoising step runs the current prompt plus generated prefix plus active block bidirectionally, then commits the most
     * confident masked positions.</p>
     */
    public Response generateDiffusion(UUID sessionId, PromptContext promptContext, GeneratorParameters parameters,
            GenerateEvent eventFired, int blockLength, float threshold) {
        Preconditions.checkArgument(blockLength > 0, "blockLength must be > 0");
        Preconditions.checkArgument(Float.isFinite(threshold) && threshold >= 0.0f, "threshold must be finite and >= 0");
        long startNanos = System.nanoTime();
        int maxNewTokens = parameters.maxTokens.or(() -> parameters.ntokens).orElse(blockLength);
        maxNewTokens = Math.max(1, maxNewTokens);
        int effectiveBlockLength = Math.min(blockLength, maxNewTokens);
        int[] promptTokens = constructPromptTokensForRuntime(promptContext.getPrompt());
        List<Integer> generated = new ArrayList<>(maxNewTokens);
        Random random = new Random(parameters.seed.orElse(42));
        float temperature = parameters.temperature.orElse(0.0f);
        int nfe = 1;
        long firstTokenNanos = 0L;
        Optional<String> previousAdapter = activeLoraAdapterId();
        clearActiveAdapter();

        try (AbstractModel.TensorPlanTraceScope ignoredTrace = openTensorPlanTrace(sessionId);
             KvCacheSession kvSession = newKvCacheSession();
             AbstractTensor promptHidden = forwardCausalWithCache(promptTokens, 0, kvSession)) {
            int nextToken = tokenFromHiddenRow(promptHidden, promptTokens.length - 1, temperature, random);
            emitToken(nextToken, generated, eventFired, startNanos);
            firstTokenNanos = System.nanoTime() - startNanos;
            if (config.eosTokens.contains(nextToken)) {
                InferenceProfiler.counter(metricRegistry, "nemotron_labs_diffusion.linear_spec.initial_seed_eos").inc();
            }
            while (generated.size() < maxNewTokens) {
                if (config.eosTokens.contains(generated.getLast())) {
                    break;
                }
                int cacheLength = kvSession.length();
                int remaining = maxNewTokens - generated.size();
                int currentBlockLength = Math.min(effectiveBlockLength, remaining);
                int[] block = new int[currentBlockLength];
                boolean[] masked = new boolean[currentBlockLength];
                for (int i = 0; i < currentBlockLength; i++) {
                    block[i] = ((NemotronLabsDiffusionConfig) config).maskTokenId;
                    masked[i] = true;
                }
                block[0] = nextToken;
                masked[0] = false;
                do {
                    nfe++;
                    enableLinearSpecDraftAdapter();
                    try (AbstractTensor hidden = forwardDenoisingBlock(kvSession, block)) {
                        int accepted = fillMaskedPositionsFromDraft(hidden, block, masked, threshold, temperature,
                                random);
                        metricRegistry.counter("nemotron_labs_diffusion.diffusion.transferred_tokens").inc(accepted);
                        metricRegistry.counter("nemotron_labs_diffusion.diffusion.mask_tokens_remaining")
                                .inc(maskedCount(masked));
                    } finally {
                        clearActiveAdapter();
                    }
                } while (threshold > 0.0f && anyMasked(masked));

                int[] verifiedTokens;
                try (AbstractTensor verifiedHidden = forwardCausalWithCache(block, cacheLength, kvSession,
                        CacheExecutionMode.VERIFY_AND_UPDATE_CACHE)) {
                    verifiedTokens = tokensFromHiddenRows(verifiedHidden, temperature, random);
                    nfe++;
                }
                int accepted = acceptedPrefixLength(block, verifiedTokens);
                accepted = Math.min(accepted, maxNewTokens - generated.size());
                int emittedAccepted = 0;
                boolean stoppedOnEos = false;
                for (int i = 0; i < accepted; i++) {
                    emitToken(verifiedTokens[i], generated, eventFired, startNanos);
                    emittedAccepted++;
                    if (config.eosTokens.contains(verifiedTokens[i])) {
                        stoppedOnEos = true;
                        InferenceProfiler.counter(metricRegistry, "nemotron_labs_diffusion.linear_spec.verify_eos").inc();
                        break;
                    }
                }
                InferenceProfiler.counter(metricRegistry, "nemotron_labs_diffusion.linear_spec.blocks").inc();
                InferenceProfiler.counter(metricRegistry, "nemotron_labs_diffusion.linear_spec.accepted_tokens")
                        .inc(emittedAccepted);
                InferenceProfiler.counter(metricRegistry, "nemotron_labs_diffusion.linear_spec.accepted_"
                        + acceptanceBucket(emittedAccepted)).inc();
                InferenceProfiler.counter(metricRegistry, "nemotron_labs_diffusion.linear_spec.block_length_"
                        + acceptanceBucket(currentBlockLength)).inc();
                kvSession.crop(cacheLength + emittedAccepted);
                nextToken = verifiedTokens[Math.max(0, emittedAccepted - 1)];
                if (stoppedOnEos || (!generated.isEmpty() && config.eosTokens.contains(generated.getLast()))) {
                    break;
                }
            }
        } finally {
            previousAdapter.ifPresent(this::setActiveAdapter);
        }

        metricRegistry.counter("nemotron_labs_diffusion.diffusion.nfe").inc(nfe);
        int[] generatedIds = generated.stream().mapToInt(Integer::intValue).toArray();
        String text = decodeTokens(generatedIds, true);
        String textWithSpecial = decodeTokens(generatedIds, false);
        long totalMs = Math.round(elapsedMs(startNanos));
        double tokensPerSecond = totalMs <= 0 ? 0.0 : generated.size() * 1000.0 / totalMs;
        LOGGER.info("nemotron_diffusion_complete prompt_tokens={} generated_tokens={} nfe={} total_ms={} tokens_per_second={}",
                promptTokens.length, generated.size(), nfe, totalMs, tokensPerSecond);
        FinishReason reason = !generated.isEmpty() && config.eosTokens.contains(generated.getLast())
                ? FinishReason.STOP_TOKEN
                : FinishReason.MAX_TOKENS;
        return timedResponse(new Response(text, textWithSpecial, reason, promptTokens.length,
                generated, 0, totalMs, List.of()), startNanos, firstTokenNanos);
    }

    private Response timedResponse(Response response, long startNanos, long firstTokenNanos) {
        double totalMs = elapsedMs(startNanos);
        double firstTokenMs = firstTokenNanos <= 0L ? 0.0 : firstTokenNanos / 1_000_000.0;
        double averageMs = response.generatedTokens.isEmpty() ? 0.0 : totalMs / response.generatedTokens.size();
        return postProcessResponse(response.copyWithTiming(firstTokenMs, averageMs, totalMs));
    }

    private void enableLinearSpecDraftAdapter() {
        if (linearSpecLoraAdapterRegistered) {
            setActiveAdapter(LINEAR_SPEC_LORA_ADAPTER_ID);
            InferenceProfiler.counter(metricRegistry, "nemotron_labs_diffusion.linear_spec.lora_draft_enabled").inc();
        }
    }

    private int acceptedPrefixLength(int[] block, int[] verifiedTokens) {
        int accepted = 0;
        for (int i = 0; i < block.length - 1; i++) {
            if (verifiedTokens[i] == block[i + 1]) {
                accepted++;
            } else {
                break;
            }
        }
        return accepted + 1;
    }

    private String acceptanceBucket(int value) {
        if (value <= 1) {
            return "1";
        }
        int upper = Integer.highestOneBit(value - 1) << 1;
        return "le_" + upper;
    }

    private void emitToken(int token, List<Integer> generated, GenerateEvent eventFired, long startNanos) {
        generated.add(token);
        String raw = decodeTokens(new int[] { token }, false);
        eventFired.emit(token, raw, raw, elapsedMs(startNanos));
    }

    private int tokenFromHiddenRow(AbstractTensor hidden, int row, float temperature, Random random) {
        try (AbstractTensor logits = logitsForHiddenRow(hidden, row);
             AbstractTensor argMax = tensorAllocator.getDirty(DType.F32, TensorShape.of(1, 2))) {
            return tokenFromLogits(logits, argMax, temperature, random).token();
        }
    }

    private int fillMaskedPositionsFromDraft(AbstractTensor hidden, int[] block, boolean[] masked, float threshold,
            float temperature, Random random) {
        int accepted = 0;
        float bestConfidence = Float.NEGATIVE_INFINITY;
        int bestPosition = -1;
        int bestToken = -1;
        try (AbstractTensor logits = logitsForHiddenRows(hidden);
             AbstractTensor argMax = tensorAllocator.getDirty(DType.F32, TensorShape.of(1, 2))) {
            for (int position = 0; position < block.length; position++) {
                if (!masked[position]) {
                    continue;
                }
                try (AbstractTensor rowLogits = logits.slice(position)) {
                    TokenConfidence tokenConfidence = tokenFromLogits(rowLogits, argMax,
                            threshold > 0.0f ? Math.max(temperature, 0.0f) : temperature, random,
                            threshold > 0.0f);
                    int token = tokenConfidence.token();
                    float confidence = tokenConfidence.confidence();
                    if (threshold == 0.0f || confidence >= threshold) {
                        block[position] = token;
                        masked[position] = false;
                        accepted++;
                    } else if (confidence > bestConfidence) {
                        bestConfidence = confidence;
                        bestPosition = position;
                        bestToken = token;
                    }
                }
            }
        }
        if (accepted == 0 && bestPosition >= 0) {
            block[bestPosition] = bestToken;
            masked[bestPosition] = false;
            accepted = 1;
        }
        return accepted;
    }

    private int[] tokensFromHiddenRows(AbstractTensor hidden, float temperature, Random random) {
        int rows = (int) hidden.shape().first();
        int[] tokens = new int[rows];
        try (AbstractTensor logits = logitsForHiddenRows(hidden);
             AbstractTensor argMax = tensorAllocator.getDirty(DType.F32, TensorShape.of(1, 2))) {
            for (int row = 0; row < rows; row++) {
                try (AbstractTensor rowLogits = logits.slice(row)) {
                    tokens[row] = tokenFromLogits(rowLogits, argMax, temperature, random).token();
                }
            }
        }
        return tokens;
    }

    private TokenConfidence tokenFromLogits(AbstractTensor logits, AbstractTensor argMax, float temperature,
            Random random) {
        return tokenFromLogits(logits, argMax, temperature, random, false);
    }

    private TokenConfidence tokenFromLogits(AbstractTensor logits, AbstractTensor argMax, float temperature,
            Random random, boolean requireConfidence) {
        if (temperature > 0.0f) {
            configurableTensorProvider.get().scale(1.0f / temperature, logits, 0, config.vocabularySize);
            configurableTensorProvider.get().softMax(logits, 0, config.vocabularySize);
            float sample = random.nextFloat();
            float cumulative = 0.0f;
            int selected = config.vocabularySize - 1;
            for (int token = 0; token < config.vocabularySize; token++) {
                cumulative += logits.get(0, token);
                if (sample <= cumulative) {
                    selected = token;
                    break;
                }
            }
            return new TokenConfidence(selected, logits.get(0, selected));
        }
        if (requireConfidence) {
            configurableTensorProvider.get().softMax(logits, 0, config.vocabularySize);
        }
        configurableTensorProvider.get().argMax(logits, argMax, 0, config.vocabularySize);
        return new TokenConfidence((int) argMax.get(0, 0), argMax.get(0, 1));
    }

    private AbstractTensor forwardDenoisingBlock(KvCacheSession kvSession, int[] block) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "nemotron_labs_diffusion.diffusion.denoise_step").time()) {
            int startPosition = kvSession.length();
            AbstractTensor current = tensorAllocator.getDirty(workingDType, TensorShape.of(block.length,
                    config.embeddingLength));
            try {
                embedTokens(block, current);
                for (int layer = 0; layer < config.numberOfLayers; layer++) {
                    AbstractTensor next = forwardLayerDenoisingNoUpdate(current, layer, startPosition, kvSession);
                    current.close();
                    current = next;
                }
                try (AbstractTensor normed = new RmsNorm(this, finalNormWeight, metricRegistry).forward(current)) {
                    current.close();
                    current = copyTensor(normed);
                }
                return current;
            } catch (RuntimeException | Error e) {
                current.close();
                throw e;
            }
        }
    }

    private AbstractTensor forwardCausalWithCache(int[] inputIds, int startPosition, KvCacheSession kvSession) {
        CacheExecutionMode mode = inputIds.length == 1
                ? CacheExecutionMode.DECODE_UPDATE_CACHE
                : CacheExecutionMode.PREFILL_UPDATE_CACHE;
        return forwardCausalWithCache(inputIds, startPosition, kvSession, mode);
    }

    private AbstractTensor forwardCausalWithCache(int[] inputIds, int startPosition, KvCacheSession kvSession,
            CacheExecutionMode mode) {
        Preconditions.checkArgument(startPosition == kvSession.length(), "startPosition must match KV session length");
        AbstractTensor current = tensorAllocator.getDirty(workingDType, TensorShape.of(inputIds.length,
                config.embeddingLength));
        try {
            embedTokens(inputIds, current);
            for (int layer = 0; layer < config.numberOfLayers; layer++) {
                AbstractTensor next = forwardLayerCausalCached(current, layer, startPosition, kvSession, mode);
                current.close();
                current = next;
            }
            try (AbstractTensor normed = new RmsNorm(this, finalNormWeight, metricRegistry).forward(current)) {
                current.close();
                current = copyTensor(normed);
            }
            kvSession.advanceLength(startPosition + inputIds.length);
            return current;
        } catch (RuntimeException | Error e) {
            current.close();
            throw e;
        }
    }

    @Override
    public boolean applyRotaryEmbedding(AbstractTensor query, AbstractTensor key, int absolutePosition,
            int queryHeads, int keyValueHeads, int headSize, TensorOperations operations) {
        rope.apply(query, key, absolutePosition, queryHeads, keyValueHeads, operations);
        return true;
    }

    private AbstractTensor forwardLayerCausalCached(AbstractTensor input, int layer, int startPosition,
            KvCacheSession kvSession, CacheExecutionMode mode) {
        TensorOperations ops = configurableTensorProvider.get();
        ForwardPhase phase = mode == CacheExecutionMode.DECODE_UPDATE_CACHE ? ForwardPhase.DECODE : ForwardPhase.PREFILL;
        try (AbstractTensor normed = new RmsNorm(this, inputNormWeights[layer], metricRegistry).forward(input);
             AbstractTensor attentionOutput = kvCacheSelfAttentions[layer].forward(normed, startPosition, kvSession,
                     mode, Optional.empty(), phase);
             AbstractTensor afterAttention = tensorAllocator.getDirty(DType.F32, input.shape())) {
            afterAttention.copyFrom(input, 0, 0, (int) input.size());
            ops.accumulate(afterAttention, attentionOutput, 0, config.embeddingLength);
            try (AbstractTensor postAttentionNorm = new RmsNorm(this, postAttentionNormWeights[layer], metricRegistry)
                         .forward(afterAttention);
                 AbstractTensor postAttentionProjectionInput = maybeQuantizeReadOnly(postAttentionNorm,
                         "transformerblock.maybe_quantize.pre_ff");
                 AbstractTensor mlp = mlpBlocks[layer].forward(postAttentionProjectionInput, Optional.empty(), phase)) {
                ops.accumulate(afterAttention, mlp, 0, config.embeddingLength);
                return copyTensor(afterAttention);
            }
        }
    }

    private AbstractTensor forwardLayerDenoisingNoUpdate(AbstractTensor input, int layer, int startPosition,
            KvCacheSession kvSession) {
        TensorOperations ops = configurableTensorProvider.get();
        ForwardPhase phase = input.shape().first() == 1 ? ForwardPhase.DECODE : ForwardPhase.PREFILL;
        try (AbstractTensor normed = new RmsNorm(this, inputNormWeights[layer], metricRegistry).forward(input);
             AbstractTensor attentionOutput = kvCacheSelfAttentions[layer].forward(normed, startPosition, kvSession,
                     CacheExecutionMode.DENOISE_BLOCK_NO_UPDATE, Optional.empty(), phase);
             AbstractTensor afterAttention = tensorAllocator.getDirty(DType.F32, input.shape())) {
            afterAttention.copyFrom(input, 0, 0, (int) input.size());
            ops.accumulate(afterAttention, attentionOutput, 0, config.embeddingLength);
            try (AbstractTensor postAttentionNorm = new RmsNorm(this, postAttentionNormWeights[layer], metricRegistry)
                         .forward(afterAttention);
                 AbstractTensor postAttentionProjectionInput = maybeQuantizeReadOnly(postAttentionNorm,
                         "transformerblock.maybe_quantize.pre_ff");
                 AbstractTensor mlp = mlpBlocks[layer].forward(postAttentionProjectionInput, Optional.empty(), phase)) {
                ops.accumulate(afterAttention, mlp, 0, config.embeddingLength);
                return copyTensor(afterAttention);
            }
        }
    }

    private AbstractTensor logitsForHiddenRows(AbstractTensor hidden) {
        AbstractTensor logits = tensorAllocator.getDirty(DType.F32,
                TensorShape.of((int) hidden.shape().first(), config.vocabularySize));
        try {
            if (diffusionHeadWeight.dType() == DType.Q4) {
                try (AbstractTensor projectionInput = maybeQuantizeReadOnly(hidden,
                        "nemotron_labs_diffusion.maybe_quantize.logits_block_projection")) {
                    if (isGpuDiffusionBlockProjectionEnabled()) {
                        gpuLogitsBlockProjection(logits, projectionInput);
                    } else {
                        project(logits, projectionInput, diffusionHeadWeight, config.embeddingLength,
                                config.vocabularySize, "nemotron_labs_diffusion.logits_block_projection");
                    }
                }
            } else {
                project(logits, hidden, diffusionHeadWeight, config.embeddingLength, config.vocabularySize,
                        "nemotron_labs_diffusion.logits_block_projection");
            }
            return logits;
        } catch (RuntimeException | Error e) {
            logits.close();
            throw e;
        }
    }

    private void gpuLogitsBlockProjection(AbstractTensor logits, AbstractTensor projectionInput) {
        TensorOperations gpu = tensorOperations(TensorProviderKind.GPU)
                .orElseThrow(() -> new IllegalStateException("GPU diffusion block projection requested, but GPU provider is unavailable"));
        Preconditions.checkArgument(projectionInput.shape().first() * (long) config.vocabularySize * DType.F32.size()
                        <= 16L * 1024L * 1024L,
                "GPU diffusion block logits exceed current scratch size: rows=%s vocab=%s",
                projectionInput.shape().first(), config.vocabularySize);
        gpu.registerModelTensor(diffusionHeadWeight);
        InferenceProfiler.counter(metricRegistry,
                "nemotron_labs_diffusion.logits_block_projection.provider_gpu").inc();
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "nemotron_labs_diffusion.logits_block_projection").time()) {
            gpu.dotProductChunk(logits, projectionInput, diffusionHeadWeight, 0, config.embeddingLength, 0,
                    config.vocabularySize);
        }
    }

    private AbstractTensor logitsForHiddenRow(AbstractTensor hidden, int row) {
        AbstractTensor logits = tensorAllocator.getDirty(DType.F32, TensorShape.of(1, config.vocabularySize));
        try (AbstractTensor hiddenRow = hidden.slice(row)) {
            if (diffusionHeadWeight.dType() == DType.Q4) {
                try (AbstractTensor projectionInput = maybeQuantizeReadOnly(hiddenRow,
                        "nemotron_labs_diffusion.maybe_quantize.logits_projection")) {
                    project(logits, projectionInput, diffusionHeadWeight, config.embeddingLength, config.vocabularySize,
                            "nemotron_labs_diffusion.logits_projection");
                }
            } else {
                project(logits, hiddenRow, diffusionHeadWeight, config.embeddingLength, config.vocabularySize,
                        "nemotron_labs_diffusion.logits_projection");
            }
            return logits;
        } catch (RuntimeException | Error e) {
            logits.close();
            throw e;
        }
    }

    private int maskedCount(boolean[] masked) {
        int count = 0;
        for (boolean value : masked) {
            if (value) {
                count++;
            }
        }
        return count;
    }

    private boolean anyMasked(boolean[] masked) {
        return maskedCount(masked) > 0;
    }

    private void embedTokens(int[] inputIds, AbstractTensor output) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "nemotron_labs_diffusion.embedding").time()) {
            for (int row = 0; row < inputIds.length; row++) {
                int token = inputIds[row];
                Preconditions.checkArgument(token >= 0 && token < embedTokenWeights.shape().first(),
                        "token out of bounds");
                if (embedTokenWeights.dType() == output.dType()) {
                    output.copyFrom(embedTokenWeights, embedTokenWeights.getOffset(token, 0), output.getOffset(row, 0),
                            config.embeddingLength);
                } else {
                    try (AbstractTensor rowView = embedTokenWeights.slice(token);
                         AbstractTensor converted = configurableTensorProvider.get().quantize(rowView, output.dType(), 0,
                                 config.embeddingLength)) {
                        output.copyFrom(converted, 0, output.getOffset(row, 0), config.embeddingLength);
                    }
                }
            }
        }
    }

    private void project(AbstractTensor output, AbstractTensor input, AbstractTensor weight, int inputLength,
            int outputLength, String metricName) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, metricName).time()) {
            for (int start = 0; start < outputLength; start += configurableTensorProvider.get().parallelSplitSize()) {
                int chunk = Math.min(configurableTensorProvider.get().parallelSplitSize(), outputLength - start);
                configurableTensorProvider.get().dotProductChunk(output, input, weight, 0, inputLength, start, chunk);
            }
        }
    }

    private AbstractTensor copyTensor(AbstractTensor source) {
        AbstractTensor copy = tensorAllocator.getDirty(source.dType(), source.shape());
        copy.copyFrom(source, 0, 0, (int) source.size());
        return copy;
    }

    private String decodeTokens(int[] tokenIds, boolean skipSpecialTokens) {
        return tokenizer.decode(new TokenIds(tokenIds), skipSpecialTokens, false, false, false);
    }

    private static float elapsedMs(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000.0f;
    }

    private record TokenConfidence(int token, float confidence) {
    }
}
