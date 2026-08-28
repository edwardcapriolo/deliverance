package io.teknek.deliverance.model.diffusiongemma;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.generator.FinishReason;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.EmbedInput;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.LayerNorm;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.generator.SampleOutput;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.grace.EncodeOptions;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.grace.TokenIds;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorMutability;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorNormalization;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.toolcallparser.ToolCallParser;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.UUID;

/**
 * Minimal DiffusionGemma checkpoint-loading and smoke-generation implementation.
 *
 * <p>This class validates the checkpoint layout and carries enough tensor execution to run the public tiny checkpoint
 * through a generation smoke. DiffusionGemma has encoder/decoder/cache semantics that differ from the existing AR models,
 * and Gemma4 is not considered a correctness baseline here; HF-parity generation still requires the remaining
 * DiffusionGemma-specific attention and denoising semantics.</p>
 */
public final class DiffusionGemmaModel extends AbstractModel {
    public record DiffusionGemmaModelOutput(AbstractTensor lastHiddenState) implements AutoCloseable {
        @Override
        public void close() {
            lastHiddenState.close();
        }
    }

    private final DiffusionGemmaConfig diffusionConfig;
    private final List<AbstractTensor> loadedRepresentativeTensors = new ArrayList<>();
    private AbstractTensor decoderEmbeddingWeights;
    private AbstractTensor outputNormWeights;
    private AbstractTensor lmHeadWeights;
    private DiffusionGemmaSelfConditioning selfConditioning;
    private AbstractTensor[] encoderInputNormWeights;
    private AbstractTensor[] encoderQueryWeights;
    private AbstractTensor[] encoderKeyWeights;
    private AbstractTensor[] encoderValueWeights;
    private AbstractTensor[] encoderOutputWeights;
    private AbstractTensor[] encoderPostAttentionNormWeights;
    private AbstractTensor[] encoderGateWeights;
    private AbstractTensor[] encoderUpWeights;
    private AbstractTensor[] encoderDownWeights;

    public DiffusionGemmaModel(InferenceType inferenceType, Config config, WeightLoader weights,
            PreTrainedTokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType,
            ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry,
            TensorAllocator tensorAllocator, KvBufferCacheSettings kvBufferCacheSettings, ToolCallParser toolCallParser,
            WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType,
                configurableTensorProvider, metricRegistry, tensorAllocator, kvBufferCacheSettings, toolCallParser, pool,
                tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
        this.diffusionConfig = (DiffusionGemmaConfig) config;
    }

    /**
     * Embeds a tensor-backed decoder canvas using {@code model.decoder.embed_tokens.weight}.
     *
     * <p>The input canvas is a F32 token-id tensor shaped {@code [batch, canvasLength]}. The returned tensor is shaped
     * {@code [batch, canvasLength, hiddenSize]} and uses the embedding tensor's dtype. This is the first text-only Phase 6
     * building block: it proves that the DiffusionGemma checkpoint can provide real decoder canvas embeddings without
     * relying on Gemma4 execution code.</p>
     */
    public AbstractTensor embedCanvasTokens(AbstractTensor canvasTokens) {
        Preconditions.checkState(decoderEmbeddingWeights != null,
                "DiffusionGemmaModel must be initialized before embedding canvas tokens");
        TensorMutability.unwrapReadOnly(canvasTokens);
        Preconditions.checkArgument(canvasTokens.dims() == 2, "canvasTokens must have shape [batch, canvasLength]");
        Preconditions.checkArgument(canvasTokens.shape().last() == diffusionConfig.canvasLength,
                "canvasTokens length must match config.canvasLength");
        Preconditions.checkArgument(canvasTokens.dType() == DType.F32, "canvasTokens must be F32 token-id tensor");

        int batchSize = (int) canvasTokens.shape().first();
        int hiddenSize = diffusionConfig.textConfig.hiddenSize;
        AbstractTensor embeddings = tensorAllocator.getDirty(decoderEmbeddingWeights.dType(),
                TensorShape.of(batchSize, diffusionConfig.canvasLength, hiddenSize));
        for (int batch = 0; batch < batchSize; batch++) {
            for (int position = 0; position < diffusionConfig.canvasLength; position++) {
                int tokenId = (int) canvasTokens.get(batch, position);
                Preconditions.checkArgument(tokenId >= 0 && tokenId < decoderEmbeddingWeights.shape().first(),
                        "canvas token id out of bounds");
                embeddings.copyFrom(decoderEmbeddingWeights, decoderEmbeddingWeights.getOffset(tokenId, 0),
                        embeddings.getOffset(batch, position, 0), hiddenSize);
            }
        }
        return embeddings;
    }

    public AbstractTensor applySelfConditioning(AbstractTensor inputsEmbeds, AbstractTensor selfConditioningSignal) {
        Preconditions.checkState(selfConditioning != null,
                "DiffusionGemmaModel must be initialized before applying self-conditioning");
        return selfConditioning.forward(inputsEmbeds, selfConditioningSignal);
    }

    /**
     * Phase 6 text-only forward skeleton.
     *
     * <p>This follows the beginning of the HF decoder path: canvas token IDs are embedded, missing self-conditioning
     * logits are represented by a zero self-conditioning signal, and the self-conditioning block produces the current
     * hidden state. Decoder attention/layers/final norm/logits are intentionally not implemented here yet.</p>
     */
    public DiffusionGemmaModelOutput forwardTextOnly(AbstractTensor decoderInputIds) {
        Preconditions.checkArgument(decoderInputIds.dims() == 2,
                "decoderInputIds must have shape [batch, canvasLength]");
        try (AbstractTensor inputEmbeddings = embedCanvasTokens(decoderInputIds);
             AbstractTensor zeroSelfConditioning = tensorAllocator.getDirty(DType.F32,
                     TensorShape.of((int) decoderInputIds.shape().first(), diffusionConfig.canvasLength,
                             diffusionConfig.textConfig.hiddenSize))) {
            zeroSelfConditioning.clear();
            return new DiffusionGemmaModelOutput(applySelfConditioning(inputEmbeddings, zeroSelfConditioning));
        }
    }

    /** Projects one hidden-state canvas position to full vocabulary logits. */
    public AbstractTensor logitsForCanvasPosition(AbstractTensor hiddenStates, int batch, int position) {
        Preconditions.checkState(lmHeadWeights != null,
                "DiffusionGemmaModel must be initialized before projecting logits");
        Preconditions.checkArgument(hiddenStates.dims() == 3, "hiddenStates must be [batch, canvasLength, hiddenSize]");
        Preconditions.checkArgument(batch >= 0 && batch < hiddenStates.shape().dim(0), "batch out of bounds");
        Preconditions.checkArgument(position >= 0 && position < hiddenStates.shape().dim(1), "position out of bounds");
        int hidden = diffusionConfig.textConfig.hiddenSize;
        int vocab = diffusionConfig.textConfig.vocabSize;
        AbstractTensor logits = tensorAllocator.getDirty(DType.F32, TensorShape.of(1, vocab));
        try (AbstractTensor flatHidden = tensorAllocator.getDirty(DType.F32, TensorShape.of(1, hidden))) {
            flatHidden.copyFrom(hiddenStates, hiddenStates.getOffset(batch, position, 0), 0, hidden);
            for (int start = 0; start < vocab; start += configurableTensorProvider.get().parallelSplitSize()) {
                int chunk = Math.min(configurableTensorProvider.get().parallelSplitSize(), vocab - start);
                configurableTensorProvider.get().dotProductChunk(logits, flatHidden, lmHeadWeights, 0, hidden, start,
                        chunk);
            }
            return logits;
        } catch (RuntimeException | Error e) {
            logits.close();
            throw e;
        }
    }

    /**
     * Projects encoder text embeddings into the read-only K/V cache consumed by the diffusion decoder.
     *
     * <p>This is a Phase 6 cache-construction slice, not a complete encoder forward. It embeds input token ids and writes
     * per-layer key/value projection rows into {@code kvBuffer}. Full-attention layers in HF may omit {@code v_proj}; in
     * that case the value states are the key states, so this method writes the projected keys to both cache streams.</p>
     */
    public void encodeTextToCache(AbstractTensor inputIds, KvBufferCache.KvBuffer kvBuffer) {
        Preconditions.checkArgument(inputIds.dims() == 2, "inputIds must be [batch, sequence]");
        Preconditions.checkState(encoderKeyWeights != null, "DiffusionGemmaModel must be initialized first");
        int batchSize = (int) inputIds.shape().first();
        int sequenceLength = (int) inputIds.shape().last();
        int hidden = diffusionConfig.textConfig.hiddenSize;
        try (AbstractTensor embeddings3d = tensorAllocator.getDirty(decoderEmbeddingWeights.dType(),
                     TensorShape.of(batchSize, sequenceLength, hidden));
             AbstractTensor embeddings2d = tensorAllocator.getDirty(decoderEmbeddingWeights.dType(),
                     TensorShape.of(batchSize * sequenceLength, hidden))) {
            embedTokenIds(inputIds, embeddings3d);
            flatten3d(embeddings3d, embeddings2d);
            int rows = batchSize * sequenceLength;
            for (int layer = 0; layer < diffusionConfig.textConfig.numHiddenLayers; layer++) {
                AbstractTensor keyWeight = encoderKeyWeights[layer];
                AbstractTensor valueWeight = encoderValueWeights[layer] == null ? keyWeight : encoderValueWeights[layer];
                int kvLength = (int) keyWeight.shape().first();
                try (AbstractTensor keys = tensorAllocator.getDirty(DType.F32, TensorShape.of(rows, kvLength));
                     AbstractTensor values = tensorAllocator.getDirty(DType.F32, TensorShape.of(rows, kvLength))) {
                    for (int start = 0; start < kvLength; start += configurableTensorProvider.get().parallelSplitSize()) {
                        int chunk = Math.min(configurableTensorProvider.get().parallelSplitSize(), kvLength - start);
                        configurableTensorProvider.get().dotProductChunk(keys, embeddings2d, keyWeight, 0, hidden, start,
                                chunk);
                        configurableTensorProvider.get().dotProductChunk(values, embeddings2d, valueWeight, 0, hidden,
                                start, chunk);
                    }
                    writeCacheRows(kvBuffer, layer, keys, values, batchSize, sequenceLength, kvLength);
                }
            }
        }
    }

    /**
     * Runs one DiffusionGemma text layer over a full text sequence.
     *
     * <p>This is the first executable encoder-layer slice. It intentionally covers the tensor data flow that the tiny
     * checkpoint can prove locally: token embedding, input RMSNorm, Q/K/V projections, full-sequence scaled-dot-product
     * attention with GQA head mapping, output projection, post-attention RMSNorm, MLP, and residuals. RoPE,
     * sliding-window masking, and decoder cross-attention are separate follow-up work.</p>
     */
    public AbstractTensor forwardTextEncoderLayer(AbstractTensor inputIds, int layerIndex) {
        Preconditions.checkArgument(inputIds.dims() == 2, "inputIds must be [batch, sequence]");
        Preconditions.checkArgument(layerIndex >= 0 && layerIndex < diffusionConfig.textConfig.numHiddenLayers,
                "layerIndex out of bounds");
        Preconditions.checkState(encoderQueryWeights != null, "DiffusionGemmaModel must be initialized first");
        Preconditions.checkState(encoderInputNormWeights[layerIndex] != null
                        && encoderOutputWeights[layerIndex] != null
                        && encoderPostAttentionNormWeights[layerIndex] != null
                        && encoderGateWeights[layerIndex] != null
                        && encoderUpWeights[layerIndex] != null
                        && encoderDownWeights[layerIndex] != null,
                "layer " + layerIndex + " is missing tensors required for encoder forward");

        int batchSize = (int) inputIds.shape().first();
        int sequenceLength = (int) inputIds.shape().last();
        int hidden = diffusionConfig.textConfig.hiddenSize;
        AbstractTensor output = tensorAllocator.getDirty(DType.F32, TensorShape.of(batchSize, sequenceLength, hidden));
        try (AbstractTensor embeddings3d = tensorAllocator.getDirty(decoderEmbeddingWeights.dType(),
                     TensorShape.of(batchSize, sequenceLength, hidden));
             AbstractTensor input2d = tensorAllocator.getDirty(DType.F32, TensorShape.of(batchSize * sequenceLength, hidden))) {
            embedTokenIds(inputIds, embeddings3d);
            flatten3d(embeddings3d, input2d);
            try (AbstractTensor layerOutput2d = forwardTextEncoderLayer2d(input2d, layerIndex, batchSize,
                         sequenceLength)) {
                inflate3d(layerOutput2d, output, batchSize, sequenceLength, hidden);
                return output;
            }
        } catch (RuntimeException | Error e) {
            output.close();
            throw e;
        }
    }

    /** Runs every currently loaded text layer over token IDs and returns final-normalized hidden states. */
    public DiffusionGemmaModelOutput forwardTextEncoder(AbstractTensor inputIds) {
        Preconditions.checkArgument(inputIds.dims() == 2, "inputIds must be [batch, sequence]");
        Preconditions.checkState(outputNormWeights != null, "DiffusionGemmaModel must be initialized first");
        int batchSize = (int) inputIds.shape().first();
        int sequenceLength = (int) inputIds.shape().last();
        int hidden = diffusionConfig.textConfig.hiddenSize;
        AbstractTensor output = tensorAllocator.getDirty(DType.F32, TensorShape.of(batchSize, sequenceLength, hidden));
        try (AbstractTensor embeddings3d = tensorAllocator.getDirty(decoderEmbeddingWeights.dType(),
                     TensorShape.of(batchSize, sequenceLength, hidden));
             AbstractTensor input2d = tensorAllocator.getDirty(DType.F32, TensorShape.of(batchSize * sequenceLength,
                     hidden))) {
            embedTokenIds(inputIds, embeddings3d);
            flatten3d(embeddings3d, input2d);
            try (AbstractTensor encoded2d = forwardHiddenThroughLoadedLayers(input2d, batchSize, sequenceLength);
                 AbstractTensor encoded3d = tensorAllocator.getDirty(DType.F32,
                         TensorShape.of(batchSize, sequenceLength, hidden))) {
                inflate3d(encoded2d, encoded3d, batchSize, sequenceLength, hidden);
                TensorNormalization.rmsNormLastDim(output, encoded3d, outputNormWeights,
                        diffusionConfig.textConfig.rmsNormEps, configurableTensorProvider.get(), pool);
                return new DiffusionGemmaModelOutput(output);
            }
        } catch (RuntimeException | Error e) {
            output.close();
            throw e;
        }
    }

    /**
     * First end-to-end DiffusionGemma smoke generation path.
     *
     * <p>This intentionally proves the Java pipeline can tokenize, build prompt cache rows, denoise a decoder canvas,
     * project logits, choose tokens, and decode a response on the tiny checkpoint. It is not yet HF-parity generation:
     * decoder cross-attention, RoPE, sliding-window masks, entropy-bound acceptance, and stopping criteria are still open.</p>
     */
    @Override
    public Response generate(UUID sessionId, PromptContext promptContext, GeneratorParameters generatorParameters,
            GenerateEvent eventFired) {
        long startNanos = System.nanoTime();
        int[] promptTokens = tokenizer.encode(promptContext.getPrompt(), EncodeOptions.defaults()).inputIds();
        if (promptTokens.length == 0) {
            promptTokens = new int[] { diffusionConfig.textConfig.bosTokenId == null ? 2 : diffusionConfig.textConfig.bosTokenId };
        }
        int requestedTokens = generatorParameters.maxTokens
                .or(() -> generatorParameters.ntokens)
                .orElse(Math.min(8, diffusionConfig.canvasLength));
        int tokensToReturn = Math.max(1, Math.min(requestedTokens, diffusionConfig.canvasLength));
        int seed = generatorParameters.seed.orElse(42);
        List<Integer> generatedTokens = new ArrayList<>(tokensToReturn);

        try (AbstractTensor promptIds = tokenTensor(promptTokens);
             KvBufferCache.KvBuffer kvBuffer = newKvBuffer();
             AbstractTensor canvas = tensorAllocator.getDirty(DType.F32, TensorShape.of(1, diffusionConfig.canvasLength));
             AbstractTensor argmax = tensorAllocator.getDirty(DType.F32, TensorShape.of(1, 2))) {
            encodeTextToCache(promptIds, kvBuffer);
            new EntropyBoundSampler(1.0f, diffusionConfig.canvasLength, diffusionConfig.textConfig.vocabSize,
                    new Random(seed), configurableTensorProvider.get(), metricRegistry).initializeCanvas(canvas);

            try (AbstractTensor hidden = forwardCanvas(canvas, null)) {
                for (int position = 0; position < tokensToReturn; position++) {
                    try (AbstractTensor logits = logitsForCanvasPosition(hidden, 0, position)) {
                        configurableTensorProvider.get().argMax(logits, argmax, 0, diffusionConfig.textConfig.vocabSize);
                        int token = (int) argmax.get(0, 0);
                        generatedTokens.add(token);
                        canvas.set(token, 0, position);
                        String raw = decodeTokens(new int[] { token }, false);
                        eventFired.emit(token, raw, raw, elapsedMs(startNanos));
                    }
                }
            }
        }

        int[] generated = generatedTokens.stream().mapToInt(Integer::intValue).toArray();
        String textWithSpecialTokens = decodeTokens(generated, false);
        String text = decodeTokens(generated, true);
        long totalMs = Math.round(elapsedMs(startNanos));
        return postProcessResponse(new Response(text, textWithSpecialTokens, FinishReason.MAX_TOKENS,
                promptTokens.length, generatedTokens, 0, totalMs, List.of()));
    }

    @Override
    protected EmbedInput loadInputWeights() {
        decoderEmbeddingWeights = loadFirstPresent("decoder embeddings",
                "model.decoder.embed_tokens.weight",
                "model.encoder.language_model.embed_tokens.weight");
        return new EmbedInput(this) {
            @Override
            public AbstractTensor inputTokenToEmbedding(int inputToken, int position) {
                throw new UnsupportedOperationException("DiffusionGemma forward is not implemented yet");
            }
        };
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        outputNormWeights = loadFirstPresent("decoder norm", "model.decoder.norm.weight");
        lmHeadWeights = loadFirstPresent("lm head", "lm_head.weight", "model.decoder.embed_tokens.weight");
        selfConditioning = new DiffusionGemmaSelfConditioning(diffusionConfig.textConfig,
                loadFirstPresent("self-conditioning pre norm", "model.decoder.self_conditioning.pre_norm.weight"),
                loadFirstPresent("self-conditioning gate", "model.decoder.self_conditioning.gate_proj.weight"),
                loadFirstPresent("self-conditioning up", "model.decoder.self_conditioning.up_proj.weight"),
                loadFirstPresent("self-conditioning down", "model.decoder.self_conditioning.down_proj.weight"),
                configurableTensorProvider.get(), tensorAllocator, pool, metricRegistry);
        return new SampleOutput() {
            @Override
            public LayerNorm getOutputLayerNorm() {
                throw new UnsupportedOperationException("DiffusionGemma output layer norm is not implemented yet");
            }

            @Override
            public AbstractTensor getOutputLogitsWeights() {
                return lmHeadWeights;
            }
        };
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        encoderInputNormWeights = new AbstractTensor[diffusionConfig.textConfig.numHiddenLayers];
        encoderQueryWeights = new AbstractTensor[diffusionConfig.textConfig.numHiddenLayers];
        encoderKeyWeights = new AbstractTensor[diffusionConfig.textConfig.numHiddenLayers];
        encoderValueWeights = new AbstractTensor[diffusionConfig.textConfig.numHiddenLayers];
        encoderOutputWeights = new AbstractTensor[diffusionConfig.textConfig.numHiddenLayers];
        encoderPostAttentionNormWeights = new AbstractTensor[diffusionConfig.textConfig.numHiddenLayers];
        encoderGateWeights = new AbstractTensor[diffusionConfig.textConfig.numHiddenLayers];
        encoderUpWeights = new AbstractTensor[diffusionConfig.textConfig.numHiddenLayers];
        encoderDownWeights = new AbstractTensor[diffusionConfig.textConfig.numHiddenLayers];
        for (int layer = 0; layer < diffusionConfig.textConfig.numHiddenLayers; layer++) {
            encoderInputNormWeights[layer] = loadLayerRepresentative(true, layer, "input_layernorm.weight");
            encoderQueryWeights[layer] = loadLayerRepresentative(true, layer, "self_attn.q_proj.weight");
            encoderKeyWeights[layer] = loadLayerRepresentative(true, layer, "self_attn.k_proj.weight");
            encoderValueWeights[layer] = loadLayerRepresentative(false, layer, "self_attn.v_proj.weight");
            encoderOutputWeights[layer] = loadLayerRepresentative(false, layer, "self_attn.o_proj.weight");
            encoderPostAttentionNormWeights[layer] = loadLayerRepresentative(false, layer,
                    "post_attention_layernorm.weight");
            encoderGateWeights[layer] = loadLayerRepresentative(false, layer, "mlp.gate_proj.weight",
                    "mlp.gate_up_proj", "experts.gate_up_proj");
            encoderUpWeights[layer] = loadLayerRepresentative(false, layer, "mlp.up_proj.weight");
            encoderDownWeights[layer] = loadLayerRepresentative(false, layer, "mlp.down_proj.weight", "experts.down_proj");
        }
        return new TransformerBlock[0];
    }

    private AbstractTensor forwardCanvas(AbstractTensor canvasTokens, AbstractTensor selfConditioningSignal) {
        Preconditions.checkArgument(canvasTokens.dims() == 2 && canvasTokens.shape().last() == diffusionConfig.canvasLength,
                "canvasTokens must be [batch, canvasLength]");
        int batchSize = (int) canvasTokens.shape().first();
        int hidden = diffusionConfig.textConfig.hiddenSize;
        AbstractTensor output = tensorAllocator.getDirty(DType.F32, TensorShape.of(batchSize, diffusionConfig.canvasLength,
                hidden));
        try (AbstractTensor inputEmbeddings = embedCanvasTokens(canvasTokens);
             AbstractTensor zeroSignal = selfConditioningSignal == null
                     ? zeroCanvasSignal(batchSize)
                     : null;
             AbstractTensor conditioned = applySelfConditioning(inputEmbeddings,
                     selfConditioningSignal == null ? zeroSignal : selfConditioningSignal);
             AbstractTensor conditioned2d = tensorAllocator.getDirty(DType.F32,
                     TensorShape.of(batchSize * diffusionConfig.canvasLength, hidden))) {
            flatten3d(conditioned, conditioned2d);
            try (AbstractTensor final2d = forwardHiddenThroughLoadedLayers(conditioned2d, batchSize,
                         diffusionConfig.canvasLength);
                 AbstractTensor final3d = tensorAllocator.getDirty(DType.F32,
                         TensorShape.of(batchSize, diffusionConfig.canvasLength, hidden))) {
                inflate3d(final2d, final3d, batchSize, diffusionConfig.canvasLength, hidden);
                TensorNormalization.rmsNormLastDim(output, final3d, outputNormWeights,
                        diffusionConfig.textConfig.rmsNormEps, configurableTensorProvider.get(), pool);
                return output;
            }
        } catch (RuntimeException | Error e) {
            output.close();
            throw e;
        }
    }

    private AbstractTensor forwardHiddenThroughLoadedLayers(AbstractTensor input2d, int batchSize, int sequenceLength) {
        AbstractTensor current = copyTensor(input2d);
        try {
            for (int layer = 0; layer < diffusionConfig.textConfig.numHiddenLayers; layer++) {
                AbstractTensor next = forwardTextEncoderLayer2d(current, layer, batchSize, sequenceLength);
                current.close();
                current = next;
            }
            return current;
        } catch (RuntimeException | Error e) {
            current.close();
            throw e;
        }
    }

    private AbstractTensor forwardTextEncoderLayer2d(AbstractTensor input, int layerIndex, int batchSize,
            int sequenceLength) {
        int hidden = diffusionConfig.textConfig.hiddenSize;
        int attentionLength = (int) encoderQueryWeights[layerIndex].shape().first();
        int kvLength = (int) encoderKeyWeights[layerIndex].shape().first();
        int headDim = diffusionConfig.textConfig.headDim;
        int numberOfHeads = diffusionConfig.textConfig.numAttentionHeads;
        int numberOfKeyValueHeads = diffusionConfig.textConfig.numKeyValueHeads;
        Preconditions.checkArgument(attentionLength == numberOfHeads * headDim,
                "q_proj rows must match numAttentionHeads * headDim");
        Preconditions.checkArgument(kvLength == numberOfKeyValueHeads * headDim,
                "k_proj rows must match numKeyValueHeads * headDim");
        Preconditions.checkArgument(numberOfHeads % numberOfKeyValueHeads == 0,
                "numAttentionHeads must be divisible by numKeyValueHeads");
        AbstractTensor valueWeight = encoderValueWeights[layerIndex] == null
                ? encoderKeyWeights[layerIndex]
                : encoderValueWeights[layerIndex];
        TensorOperations ops = configurableTensorProvider.get();

        try (AbstractTensor normed = tensorAllocator.getDirty(DType.F32, input.shape());
             AbstractTensor query = tensorAllocator.getDirty(DType.F32, TensorShape.of(batchSize * sequenceLength,
                     attentionLength));
             AbstractTensor key = tensorAllocator.getDirty(DType.F32, TensorShape.of(batchSize * sequenceLength,
                     kvLength));
             AbstractTensor value = tensorAllocator.getDirty(DType.F32, TensorShape.of(batchSize * sequenceLength,
                     kvLength));
             AbstractTensor attended = tensorAllocator.getDirty(DType.F32, TensorShape.of(batchSize * sequenceLength,
                     attentionLength));
             AbstractTensor attentionOutput = tensorAllocator.getDirty(DType.F32, input.shape());
             AbstractTensor residualAfterAttention = tensorAllocator.getDirty(DType.F32, input.shape());
             AbstractTensor postAttentionNorm = tensorAllocator.getDirty(DType.F32, input.shape())) {
            TensorNormalization.rmsNorm(normed, input, encoderInputNormWeights[layerIndex],
                    diffusionConfig.textConfig.rmsNormEps, ops, pool);
            project(query, normed, encoderQueryWeights[layerIndex], hidden, attentionLength);
            project(key, normed, encoderKeyWeights[layerIndex], hidden, kvLength);
            project(value, normed, valueWeight, hidden, kvLength);
            fullSequenceAttention(attended, query, key, value, batchSize, sequenceLength, attentionLength, kvLength);
            project(attentionOutput, attended, encoderOutputWeights[layerIndex], attentionLength, hidden);

            residualAfterAttention.copyFrom(input, 0, 0, (int) input.size());
            ops.accumulate(residualAfterAttention, attentionOutput, 0, hidden);
            TensorNormalization.rmsNorm(postAttentionNorm, residualAfterAttention,
                    encoderPostAttentionNormWeights[layerIndex], diffusionConfig.textConfig.rmsNormEps, ops, pool);
            try (AbstractTensor mlpOutput = mlpForward(postAttentionNorm, layerIndex)) {
                ops.accumulate(residualAfterAttention, mlpOutput, 0, hidden);
                return copyTensor(residualAfterAttention);
            }
        }
    }

    private AbstractTensor mlpForward(AbstractTensor input, int layerIndex) {
        int hidden = diffusionConfig.textConfig.hiddenSize;
        int intermediate = diffusionConfig.textConfig.intermediateSize;
        TensorOperations ops = configurableTensorProvider.get();
        try (AbstractTensor gate = tensorAllocator.getDirty(DType.F32, TensorShape.of((int) input.shape().first(),
                     intermediate));
             AbstractTensor up = tensorAllocator.getDirty(DType.F32, TensorShape.of((int) input.shape().first(),
                     intermediate))) {
            project(gate, input, encoderGateWeights[layerIndex], hidden, intermediate);
            project(up, input, encoderUpWeights[layerIndex], hidden, intermediate);
            try (AbstractTensor activated = ops.activationMultiplyQuantize(gate, up,
                         diffusionConfig.textConfig.hiddenActivation, DType.F32, 0, intermediate)) {
                AbstractTensor output = tensorAllocator.getDirty(DType.F32, input.shape());
                try {
                    project(output, activated, encoderDownWeights[layerIndex], intermediate, hidden);
                    return output;
                } catch (RuntimeException | Error e) {
                    output.close();
                    throw e;
                }
            }
        }
    }

    private void fullSequenceAttention(AbstractTensor output, AbstractTensor query, AbstractTensor key,
            AbstractTensor value, int batchSize, int sequenceLength, int attentionLength, int kvLength) {
        TensorOperations ops = configurableTensorProvider.get();
        int headDim = diffusionConfig.textConfig.headDim;
        int numberOfHeads = diffusionConfig.textConfig.numAttentionHeads;
        int numberOfKeyValueHeads = diffusionConfig.textConfig.numKeyValueHeads;
        int headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        float scale = (float) (1.0 / StrictMath.sqrt(headDim));
        output.clear();
        for (int batch = 0; batch < batchSize; batch++) {
            try (AbstractTensor queryBatch = tensorAllocator.getDirty(DType.F32,
                         TensorShape.of(sequenceLength, attentionLength));
                 AbstractTensor keyBatch = tensorAllocator.getDirty(DType.F32,
                         TensorShape.of(sequenceLength, kvLength));
                 AbstractTensor valueBatch = tensorAllocator.getDirty(DType.F32,
                         TensorShape.of(sequenceLength, kvLength));
                 AbstractTensor scores = tensorAllocator.getDirty(DType.F32, TensorShape.of(1, sequenceLength))) {
                copyBatchRows(query, queryBatch, batch, sequenceLength, attentionLength);
                copyBatchRows(key, keyBatch, batch, sequenceLength, kvLength);
                copyBatchRows(value, valueBatch, batch, sequenceLength, kvLength);
                for (int row = 0; row < sequenceLength; row++) {
                    try (AbstractTensor queryRow = queryBatch.slice(row);
                         AbstractTensor outputRow = output.slice(batch * sequenceLength + row)) {
                        for (int head = 0; head < numberOfHeads; head++) {
                            int kvHead = head / headGroupSize;
                            int queryOffset = head * headDim;
                            int kvOffset = kvHead * headDim;
                            for (int keyPosition = 0; keyPosition < sequenceLength; keyPosition++) {
                                try (AbstractTensor keyRow = keyBatch.slice(keyPosition)) {
                                    scores.set(ops.dotProduct(queryRow, keyRow, queryOffset, kvOffset, headDim), 0,
                                            keyPosition);
                                }
                            }
                            ops.scaledSoftMax(scores, 0, sequenceLength, scale, null);
                            ops.saxpy(scores, valueBatch, outputRow, kvOffset, queryOffset, headDim, 0, 0,
                                    sequenceLength);
                        }
                    }
                }
            }
        }
    }

    private void project(AbstractTensor output, AbstractTensor input, AbstractTensor weight, int inputLength,
            int outputLength) {
        for (int start = 0; start < outputLength; start += configurableTensorProvider.get().parallelSplitSize()) {
            int chunk = Math.min(configurableTensorProvider.get().parallelSplitSize(), outputLength - start);
            configurableTensorProvider.get().dotProductChunk(output, input, weight, 0, inputLength, start, chunk);
        }
    }

    private void copyBatchRows(AbstractTensor source, AbstractTensor target, int batch, int rows, int rowLength) {
        int sourceStart = batch * rows;
        for (int row = 0; row < rows; row++) {
            target.copyFrom(source, source.getOffset(sourceStart + row, 0), target.getOffset(row, 0), rowLength);
        }
    }

    private void inflate3d(AbstractTensor source, AbstractTensor target, int batchSize, int rows, int rowLength) {
        for (int batch = 0; batch < batchSize; batch++) {
            for (int row = 0; row < rows; row++) {
                int flatRow = batch * rows + row;
                target.copyFrom(source, source.getOffset(flatRow, 0), target.getOffset(batch, row, 0), rowLength);
            }
        }
    }

    private AbstractTensor copyTensor(AbstractTensor source) {
        AbstractTensor copy = tensorAllocator.getDirty(source.dType(), source.shape());
        copy.copyFrom(source, 0, 0, (int) source.size());
        return copy;
    }

    private AbstractTensor tokenTensor(int[] tokenIds) {
        AbstractTensor tensor = tensorAllocator.getDirty(DType.F32, TensorShape.of(1, tokenIds.length));
        for (int i = 0; i < tokenIds.length; i++) {
            tensor.set(tokenIds[i], 0, i);
        }
        return tensor;
    }

    private AbstractTensor zeroCanvasSignal(int batchSize) {
        AbstractTensor signal = tensorAllocator.getDirty(DType.F32,
                TensorShape.of(batchSize, diffusionConfig.canvasLength, diffusionConfig.textConfig.hiddenSize));
        signal.clear();
        return signal;
    }

    private String decodeTokens(int[] tokenIds, boolean skipSpecialTokens) {
        return tokenizer.decode(new TokenIds(tokenIds), skipSpecialTokens, false, false, false);
    }

    private static float elapsedMs(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000.0f;
    }

    private AbstractTensor loadLayerRepresentative(boolean required, int layer, String... suffixes) {
        List<String> candidates = new ArrayList<>();
        for (String suffix : suffixes) {
            candidates.add("model.decoder.layers." + layer + "." + suffix);
            candidates.add("model.encoder.language_model.layers." + layer + "." + suffix);
        }
        if (required) {
            return loadFirstPresent("layer " + layer + " representative " + String.join("/", suffixes),
                    candidates.toArray(String[]::new));
        }
        return loadFirstPresentOptional(candidates.toArray(String[]::new));
    }

    private AbstractTensor loadFirstPresent(String description, String... names) {
        for (String name : names) {
            if (weights.isWeightPresent(name)) {
                AbstractTensor tensor = weights.load(name);
                loadedRepresentativeTensors.add(tensor);
                registerModelLineageTensor(name, tensor);
                configurableTensorProvider.get().registerModelTensor(tensor);
                return tensor;
            }
        }
        throw new IllegalStateException("DiffusionGemma checkpoint missing " + description + ": "
                + String.join(", ", names));
    }

    private AbstractTensor loadFirstPresentOptional(String... names) {
        for (String name : names) {
            if (weights.isWeightPresent(name)) {
                AbstractTensor tensor = weights.load(name);
                loadedRepresentativeTensors.add(tensor);
                registerModelLineageTensor(name, tensor);
                configurableTensorProvider.get().registerModelTensor(tensor);
                return tensor;
            }
        }
        return null;
    }

    private void embedTokenIds(AbstractTensor tokenIds, AbstractTensor output) {
        int hidden = diffusionConfig.textConfig.hiddenSize;
        for (int batch = 0; batch < tokenIds.shape().first(); batch++) {
            for (int position = 0; position < tokenIds.shape().last(); position++) {
                int tokenId = (int) tokenIds.get(batch, position);
                Preconditions.checkArgument(tokenId >= 0 && tokenId < decoderEmbeddingWeights.shape().first(),
                        "encoder token id out of bounds");
                output.copyFrom(decoderEmbeddingWeights, decoderEmbeddingWeights.getOffset(tokenId, 0),
                        output.getOffset(batch, position, 0), hidden);
            }
        }
    }

    private void flatten3d(AbstractTensor source, AbstractTensor target) {
        int hidden = (int) source.shape().dim(2);
        int rows = (int) (source.shape().dim(0) * source.shape().dim(1));
        if (source.dType() == target.dType()) {
            flatten3dSameDType(source, target, hidden);
            return;
        }
        try (AbstractTensor flatSource = tensorAllocator.getDirty(source.dType(), TensorShape.of(rows, hidden))) {
            flatten3dSameDType(source, flatSource, hidden);
            try (AbstractTensor converted = configurableTensorProvider.get().quantize(flatSource, target.dType(), 0,
                         hidden)) {
                target.copyFrom(converted, 0, 0, (int) converted.size());
            }
        }
    }

    private void flatten3dSameDType(AbstractTensor source, AbstractTensor target, int hidden) {
        int row = 0;
        for (int batch = 0; batch < source.shape().dim(0); batch++) {
            for (int position = 0; position < source.shape().dim(1); position++) {
                target.copyFrom(source, source.getOffset(batch, position, 0), target.getOffset(row, 0), hidden);
                row++;
            }
        }
    }

    private void writeCacheRows(KvBufferCache.KvBuffer kvBuffer, int layer, AbstractTensor keys, AbstractTensor values,
            int batchSize, int sequenceLength, int kvLength) {
        for (int batch = 0; batch < batchSize; batch++) {
            for (int position = 0; position < sequenceLength; position++) {
                int row = batch * sequenceLength + position;
                int cachePosition = batch * sequenceLength + position;
                try (AbstractTensor keyRow = kvBuffer.getKeyTensorForPosition(layer, cachePosition);
                     AbstractTensor valueRow = kvBuffer.getValTensorForPosition(layer, cachePosition)) {
                    copyRowConverting(keys, row, keyRow, kvLength);
                    copyRowConverting(values, row, valueRow, kvLength);
                }
            }
        }
    }

    private void copyRowConverting(AbstractTensor source, int sourceRow, AbstractTensor destination, int length) {
        if (source.dType() == destination.dType()) {
            destination.copyFrom(source, source.getOffset(sourceRow, 0), 0, length);
            return;
        }
        try (AbstractTensor sourceRowView = source.slice(sourceRow);
             AbstractTensor converted = configurableTensorProvider.get().quantize(sourceRowView, destination.dType(), 0,
                     length)) {
            destination.copyFrom(converted, 0, 0, length);
        }
    }

    @Override
    public void close() {
        RuntimeException failure = null;
        for (AbstractTensor tensor : loadedRepresentativeTensors) {
            try {
                tensor.close();
            } catch (RuntimeException e) {
                failure = e;
            }
        }
        loadedRepresentativeTensors.clear();
        super.close();
        if (failure != null) {
            throw failure;
        }
    }
}
