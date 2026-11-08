package io.teknek.reckoning.model.nemotronlabsdiffusion;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.generator.EmbedInput;
import io.teknek.reckoning.generator.FinishReason;
import io.teknek.reckoning.generator.ForwardPhase;
import io.teknek.reckoning.generator.GeneratorParameters;
import io.teknek.reckoning.generator.Response;
import io.teknek.reckoning.generator2.MLPBlock2;
import io.teknek.reckoning.generator2.RmsNorm2;
import io.teknek.reckoning.generator2.SampleOutputRef;
import io.teknek.reckoning.generator2.TransformerBlock2;
import io.teknek.reckoning.grace.TokenIds;
import io.teknek.reckoning.grace.PreTrainedTokenizer;
import io.teknek.reckoning.math.WrappedForkJoinPool;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.GenerateEvent;
import io.teknek.reckoning.model.InferenceProfiler;
import io.teknek.reckoning.model.llama.LlamaModel;
import io.teknek.reckoning.model.tensorparallel.TensorParallelCollectives;
import io.teknek.reckoning.model.tensorparallel.TensorParallelContext;
import io.teknek.reckoning.safetensors.Config;
import io.teknek.reckoning.safetensors.WeightLoader;
import io.teknek.reckoning.safetensors.fetch.LoraAdapterModelFetcher;
import io.teknek.reckoning.safetensors.prompt.PromptContext;
import io.teknek.reckoning.tensor.KvBufferCacheSettings;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.tensor.kv.CacheExecutionMode;
import io.teknek.reckoning.tensor.kv.KvCacheSession;
import io.teknek.reckoning.tensor2.Scale;
import io.teknek.reckoning.tensor2.ScaledSoftMax;
import io.teknek.reckoning.tensor2.TensorRef;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.ArgMax;
import io.teknek.reckoning.toolcallparser.ToolCallParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.UUID;
import java.util.stream.IntStream;


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

    private volatile TensorRef embedTokenWeightsRef;
    private TensorRef finalNormWeightRef;
    private TensorRef diffusionHeadWeightRef;
    private final NemotronLabsDiffusionRope rope;
    private boolean linearSpecLoraAdapterRegistered;

    public NemotronLabsDiffusionModel(InferenceType inferenceType, Config config, WeightLoader weights,
            PreTrainedTokenizer tokenizer, DType workingMemoryDType, DType workingMemoryQType,
            Optional<DType> modelQType,
            MetricRegistry metricRegistry, KvBufferCacheSettings kvBufferCacheSettings,
            ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization, Lighter lighter) {
        super(inferenceType, config, weights, tokenizer, workingMemoryDType, workingMemoryQType, modelQType,
                metricRegistry, kvBufferCacheSettings, toolCallParser, pool,
                tensorParallelContext, tensorParallelCollectives, outputHeadQuantization, lighter);
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
        embedTokenWeightsRef = registerModelTensorRef(weights.loadRef("encoder.embed_tokens.weight"));
        return new EmbedInput(this) {
            @Override
            public TensorRef inputTokenToEmbedding(int inputToken, int unused) {
                Preconditions.checkArgument(inputToken >= 0 && inputToken < embedTokenWeightsRef.shape().first(),
                        "input token out of bounds");
                try (TensorRef row = embedTokenWeightsRef.slice(inputToken)) {
                    TensorRef embedding = makeDenseTensorRef(1, config.embeddingLength);
                    try {
                        if (row.dType() == embedding.dType()) {
                            lighter.copy(row, 0, embedding, 0, config.embeddingLength);
                        } else {
                            try (TensorRef converted = lighter.reshape(row, embedding.dType())) {
                                lighter.copy(converted, 0, embedding, 0, config.embeddingLength);
                            }
                        }
                        return embedding;
                    } catch (RuntimeException | Error e) {
                        embedding.close();
                        throw e;
                    }
                }
            }
        };
    }

    @Override
    protected boolean usesTensorRefExecution() {
        return true;
    }

    @Override
    public boolean usesKvCache2Generation() {
        return usesTensorRefExecution();
    }

    @Override
    protected TransformerBlock2[] loadTransformerBlockWeights2() {
        DType qType = modelQType.orElse(this.modelDType);
        TransformerBlock2[] blocks = new TransformerBlock2[config.numberOfLayers];
        IntStream.range(0, config.numberOfLayers).parallel().forEach(layer -> {
            String base = "encoder.layers." + layer + ".";
            String attentionPrefix = base + "self_attn.";
            String qName = attentionPrefix + "q_proj.weight";
            String kName = attentionPrefix + "k_proj.weight";
            String vName = attentionPrefix + "v_proj.weight";
            String oName = attentionPrefix + "o_proj.weight";
            NemotronLabsDiffusionKvCacheSelfAttention2 attention =
                    new NemotronLabsDiffusionKvCacheSelfAttention2(this, layer,
                            registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(qName, weights::loadRef, qType)),
                            registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(kName, weights::loadRef, qType)),
                            registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(vName, weights::loadRef, qType)),
                            registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(oName, weights::loadRef, qType)),
                            lighter, metricRegistry, qName, kName, vName, oName, rope);
            String mlpPrefix = base + "mlp.";
            String gateName = mlpPrefix + "gate_proj.weight";
            String upName = mlpPrefix + "up_proj.weight";
            String downName = mlpPrefix + "down_proj.weight";
            MLPBlock2 mlp = new MLPBlock2(this,
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(gateName, weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(upName, weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(downName, weights::loadRef, qType)),
                    lighter, gateName, upName, downName);
            String inputNormName = base + "input_layernorm.weight";
            String postAttentionNormName = base + "post_attention_layernorm.weight";
            blocks[layer] = new TransformerBlock2(this, layer,
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                            inputNormName, weights::loadRef, qType)), 0.0f)), attention, Optional.empty(),
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                            postAttentionNormName, weights::loadRef, qType)), 0.0f)), mlp, Optional.empty(),
                    Optional.empty());
        });
        return blocks;
    }

    @Override
    protected SampleOutputRef loadOutputWeightsRef() {
        DType qType = modelQType.orElse(this.modelDType);
        finalNormWeightRef = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                "encoder.norm.weight", weights::loadRef, qType));
        TensorRef head = weights.loadRef("diffusion_head.weight");
        if (outputHeadQuantization.isPresent() && lighter.shouldQuantizeForEfficiency(head,
                outputHeadQuantization.get())) {
            TensorRef quantized = lighter.reshape(head, outputHeadQuantization.get());
            head.close();
            head = quantized;
        }
        diffusionHeadWeightRef = registerModelTensorRef(head);
        return new SampleOutputRef() {
            @Override
            public io.teknek.reckoning.generator2.LayerNorm2 outputLayerNorm() {
                return new RmsNorm2(NemotronLabsDiffusionModel.this, finalNormWeightRef, 0.0f);
            }

            @Override
            public TensorRef outputLogitsWeights() {
                return diffusionHeadWeightRef;
            }
        };
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
             TensorRef argMax = lighter.allocate(DType.F32, TensorShape.of(1, 2))) {
            try (TensorRef promptHidden = batchForwardRef(promptTokens, 0, kvSession);
                 TensorRef logits = logitsForHiddenRowRef(promptHidden, (int) promptHidden.shape().first() - 1)) {
                lighter.argMax(new ArgMax(logits).into(argMax).offsetAndLength(0, config.vocabularySize));
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
                try (TensorRef hidden = forwardRef(previousToken, position, kvSession);
                     TensorRef logits = logitsForHiddenRowRef(hidden, 0)) {
                    lighter.argMax(new ArgMax(logits).into(argMax).offsetAndLength(0, config.vocabularySize));
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
             TensorRef promptHidden = forwardRefWithCache(promptTokens, 0, kvSession,
                     CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
             int nextToken = tokenFromHiddenRowRef(promptHidden, promptTokens.length - 1, temperature, random);
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
                    try (TensorRef hidden = forwardRefWithCache(block, cacheLength, kvSession,
                            CacheExecutionMode.DENOISE_BLOCK_NO_UPDATE)) {
                        int accepted = fillMaskedPositionsFromDraftRef(hidden, block, masked, threshold, temperature,
                                random);
                        metricRegistry.counter("nemotron_labs_diffusion.diffusion.transferred_tokens").inc(accepted);
                        metricRegistry.counter("nemotron_labs_diffusion.diffusion.mask_tokens_remaining")
                                .inc(maskedCount(masked));
                    } finally {
                        clearActiveAdapter();
                    }
                } while (threshold > 0.0f && anyMasked(masked));

                int[] verifiedTokens;
                try (TensorRef verifiedHidden = forwardRefWithCache(block, cacheLength, kvSession,
                        CacheExecutionMode.VERIFY_AND_UPDATE_CACHE)) {
                    verifiedTokens = tokensFromHiddenRowsRef(verifiedHidden, temperature, random);
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

    private int tokenFromHiddenRowRef(TensorRef hidden, int row, float temperature, Random random) {
        try (TensorRef logits = logitsForHiddenRowRef(hidden, row);
             TensorRef argMax = getLighter().allocate(DType.F32, TensorShape.of(1, 2))) {
            return tokenFromLogitsRef(logits, argMax, temperature, random, false).token();
        }
    }

    private int fillMaskedPositionsFromDraftRef(TensorRef hidden, int[] block, boolean[] masked, float threshold,
            float temperature, Random random) {
        int accepted = 0;
        float bestConfidence = Float.NEGATIVE_INFINITY;
        int bestPosition = -1;
        int bestToken = -1;
        try (TensorRef logits = logitsForHiddenRowsRef(hidden);
             TensorRef argMax = getLighter().allocate(DType.F32, TensorShape.of(1, 2))) {
            for (int position = 0; position < block.length; position++) {
                if (!masked[position]) continue;
                try (TensorRef rowLogits = logits.slice(position)) {
                    TokenConfidence confidence = tokenFromLogitsRef(rowLogits, argMax,
                            threshold > 0.0f ? Math.max(temperature, 0.0f) : temperature, random, threshold > 0.0f);
                    if (threshold == 0.0f || confidence.confidence() >= threshold) {
                        block[position] = confidence.token();
                        masked[position] = false;
                        accepted++;
                    } else if (confidence.confidence() > bestConfidence) {
                        bestConfidence = confidence.confidence();
                        bestPosition = position;
                        bestToken = confidence.token();
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

    private int[] tokensFromHiddenRowsRef(TensorRef hidden, float temperature, Random random) {
        int rows = (int) hidden.shape().first();
        int[] tokens = new int[rows];
        try (TensorRef logits = logitsForHiddenRowsRef(hidden);
             TensorRef argMax = getLighter().allocate(DType.F32, TensorShape.of(1, 2))) {
            for (int row = 0; row < rows; row++) {
                try (TensorRef rowLogits = logits.slice(row)) {
                    tokens[row] = tokenFromLogitsRef(rowLogits, argMax, temperature, random, false).token();
                }
            }
        }
        return tokens;
    }

    private TokenConfidence tokenFromLogitsRef(TensorRef logits, TensorRef argMax, float temperature,
            Random random, boolean requireConfidence) {
        if (temperature > 0.0f) {
            getLighter().scale(new Scale(1.0f / temperature).target(logits)
                    .offsetAndLength(0, config.vocabularySize));
            getCompositeOps().scaledSoftMax(new ScaledSoftMax(1.0f).target(logits)
                    .offsetAndLength(0, config.vocabularySize));
            float sample = random.nextFloat();
            float cumulative = 0.0f;
            int selected = config.vocabularySize - 1;
            for (int token = 0; token < config.vocabularySize; token++) {
                cumulative += logits.get(0, token);
                if (sample <= cumulative) { selected = token; break; }
            }
            return new TokenConfidence(selected, logits.get(0, selected));
        }
        if (requireConfidence) {
            getCompositeOps().scaledSoftMax(new ScaledSoftMax(1.0f).target(logits)
                    .offsetAndLength(0, config.vocabularySize));
        }
        getLighter().argMax(new ArgMax(logits).into(argMax).offsetAndLength(0, config.vocabularySize));
        return new TokenConfidence((int) argMax.get(0, 0), argMax.get(0, 1));
    }

    private TensorRef forwardRefWithCache(int[] inputIds, int startPosition, KvCacheSession kvSession,
            CacheExecutionMode mode) {
        Preconditions.checkArgument(startPosition == kvSession.length(),
                "startPosition must match KV session length");
        TensorRef current = makeDenseTensorRef(inputIds.length, config.embeddingLength);
        try {
            embedTokensRef(inputIds, current);
            ForwardPhase phase = inputIds.length == 1 ? ForwardPhase.DECODE : ForwardPhase.PREFILL;
            for (int layer = 0; layer < config.numberOfLayers; layer++) {
                TensorRef next = transformerBlocks2[layer].forward(current, startPosition, kvSession,
                        Optional.empty(), phase, mode);
                current.close();
                current = next;
            }
            TensorRef normalized = new RmsNorm2(this, finalNormWeightRef, 0.0f).forward(current);
            current.close();
            if (mode != CacheExecutionMode.DENOISE_BLOCK_NO_UPDATE) {
                kvSession.advanceLength(startPosition + inputIds.length);
            }
            return normalized;
        } catch (RuntimeException | Error e) {
            current.close();
            throw e;
        }
    }

    private TensorRef logitsForHiddenRowRef(TensorRef hidden, int row) {
        try (TensorRef hiddenRow = hidden.slice(row);
             TensorRef normalized = new RmsNorm2(this, finalNormWeightRef, 0.0f).forward(hiddenRow)) {
            TensorRef logits = makeDenseTensorRef(1, config.vocabularySize);
            TensorRef projectionInput = normalized;
            boolean closeProjectionInput = false;
            try {
                if (projectionInput.dType() != diffusionHeadWeightRef.dType()
                        && diffusionHeadWeightRef.dType() == DType.Q4) {
                    projectionInput = lighter.reshape(projectionInput, getWorkingQType());
                    closeProjectionInput = true;
                }
                lighter.dotProductRows(logits, projectionInput, diffusionHeadWeightRef, 0, config.embeddingLength,
                        0, config.vocabularySize, 0);
                return logits;
            } catch (RuntimeException | Error e) {
                logits.close();
                throw e;
            } finally {
                if (closeProjectionInput) {
                    projectionInput.close();
                }
            }
        }
    }

    private TensorRef logitsForHiddenRowsRef(TensorRef hidden) {
        TensorRef normalized = new RmsNorm2(this, finalNormWeightRef, 0.0f).forward(hidden);
        TensorRef logits = makeDenseTensorRef((int) hidden.shape().first(), config.vocabularySize);
        TensorRef projectionInput = normalized;
        boolean closeProjectionInput = false;
        try {
            if (projectionInput.dType() != diffusionHeadWeightRef.dType() && diffusionHeadWeightRef.dType() == DType.Q4) {
                projectionInput = getLighter().reshape(projectionInput, getWorkingQType());
                closeProjectionInput = true;
            }
            getLighter().batchDotProduct(new io.teknek.reckoning.tensor2.BatchDotProduct()
                    .result(logits).a(projectionInput).b(diffusionHeadWeightRef)
                    .aColumnOffset(0).bColumnOffset(0).columnLength(config.embeddingLength)
                    .resultRowOffset(0).bRowOffset(0).rowChunkSize(config.vocabularySize)
                    .threading(io.teknek.reckoning.tensor2.Threading.NONE));
            return logits;
        } catch (RuntimeException | Error e) {
            logits.close();
            throw e;
        } finally {
            normalized.close();
            if (closeProjectionInput) projectionInput.close();
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

    private void embedTokensRef(int[] inputIds, TensorRef output) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "nemotron_labs_diffusion.embedding").time()) {
            for (int row = 0; row < inputIds.length; row++) {
                int token = inputIds[row];
                Preconditions.checkArgument(token >= 0 && token < embedTokenWeightsRef.shape().first(),
                        "token out of bounds");
                try (TensorRef source = embedTokenWeightsRef.slice(token)) {
                    if (source.dType() == output.dType()) {
                        getLighter().copy(source, 0, output, row * config.embeddingLength, config.embeddingLength);
                    } else {
                        try (TensorRef converted = getLighter().reshape(source, output.dType())) {
                            getLighter().copy(converted, 0, output, row * config.embeddingLength,
                                    config.embeddingLength);
                        }
                    }
                }
            }
        }
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
