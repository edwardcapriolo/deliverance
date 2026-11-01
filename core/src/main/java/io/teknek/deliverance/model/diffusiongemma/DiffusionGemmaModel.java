package io.teknek.deliverance.model.diffusiongemma;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.generator.FinishReason;
import io.teknek.deliverance.generator.SampleOutput;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.EmbedInput;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.generator2.SampleOutputRef;
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
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.ScaledSoftMax;
import io.teknek.deliverance.tensor2.GemmDit;
import io.teknek.deliverance.tensor2.ArgMax;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor.kv.CacheExecutionMode;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor.kv.KvWriteCursor;
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
    public record DiffusionGemmaModelOutput(TensorRef lastHiddenState) implements AutoCloseable {
        @Override public void close() { lastHiddenState.close(); }
    }

    private final DiffusionGemmaConfig diffusionConfig;
    private TensorRef decoderEmbeddingWeightsRef;
    private TensorRef lmHeadWeightsRef;
    private TensorRef selfConditioningPreNormRef;
    private TensorRef selfConditioningGateRef;
    private TensorRef selfConditioningUpRef;
    private TensorRef selfConditioningDownRef;
    private TensorRef[] encoderKeyWeightsRef;
    private TensorRef[] encoderValueWeightsRef;
    private DiffusionGemmaSelfConditioning selfConditioningRef;

    /** TensorRef-native constructor used by the active DiffusionGemma runtime. */
    public DiffusionGemmaModel(InferenceType inferenceType, Config config, WeightLoader weights,
            PreTrainedTokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType,
            MetricRegistry metricRegistry, KvBufferCacheSettings kvBufferCacheSettings, ToolCallParser toolCallParser,
            WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization,
            Lighter lighter) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType, metricRegistry,
                kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext, tensorParallelCollectives,
                outputHeadQuantization, lighter);
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
    /** Embeds canvas token IDs using tensor2 storage and operations. */
    public TensorRef embedCanvasTokensRef(TensorRef canvasTokens) {
        ensureTensorRefWeights();
        Preconditions.checkArgument(canvasTokens.dims() == 2, "canvasTokens must have shape [batch, canvasLength]");
        Preconditions.checkArgument(canvasTokens.shape().last() == diffusionConfig.canvasLength,
                "canvasTokens length must match config.canvasLength");
        TensorRef embeddings = lighter.allocate(DType.F32, TensorShape.of(canvasTokens.shape().first(),
                diffusionConfig.canvasLength, diffusionConfig.textConfig.hiddenSize));
        try {
            for (int batch = 0; batch < canvasTokens.shape().first(); batch++) {
                for (int position = 0; position < diffusionConfig.canvasLength; position++) {
                    int token = (int) canvasTokens.get(batch, position);
                    Preconditions.checkArgument(token >= 0 && token < decoderEmbeddingWeightsRef.shape().first(),
                            "canvas token id out of bounds");
                    for (int column = 0; column < diffusionConfig.textConfig.hiddenSize; column++) {
                        embeddings.set(decoderEmbeddingWeightsRef.get(token, column), batch, position, column);
                    }
                }
            }
            return embeddings;
        } catch (RuntimeException | Error error) {
            embeddings.close();
            throw error;
        }
    }

    public TensorRef applySelfConditioningRef(TensorRef inputsEmbeds, TensorRef selfConditioningSignal) {
        ensureTensorRefWeights();
        return selfConditioningRef.forward(inputsEmbeds, selfConditioningSignal);
    }

    /** TensorRef equivalent of the tested single canvas-position output projection. */
    public TensorRef logitsForCanvasPositionRef(TensorRef hiddenStates, int batch, int position) {
        ensureTensorRefWeights();
        Preconditions.checkArgument(hiddenStates.dims() == 3, "hiddenStates must be [batch, canvasLength, hiddenSize]");
        TensorRef row = lighter.allocate(DType.F32, TensorShape.of(1, diffusionConfig.textConfig.hiddenSize));
        TensorRef logits = lighter.allocate(DType.F32, TensorShape.of(1, diffusionConfig.textConfig.vocabSize));
        try {
            for (int column = 0; column < row.shape().last(); column++) {
                row.set(hiddenStates.get(batch, position, column), 0, column);
            }
            GemmDit.dotProductRows(lighter, this, "diffusiongemma.logits", logits, row, lmHeadWeightsRef,
                    0, 0, row.shape().last(), 0, logits.shape().last(), 0);
            return logits;
        } catch (RuntimeException | Error error) {
            logits.close();
            throw error;
        } finally {
            row.close();
        }
    }

    /** TensorRef self-conditioning boundary; decoder attention remains intentionally out of scope. */
    public TensorRef forwardTextOnlyRef(TensorRef decoderInputIds) {
        try (TensorRef embeddings = embedCanvasTokensRef(decoderInputIds);
             TensorRef zero = lighter.allocate(DType.F32, embeddings.shape())) {
            lighter.clear(zero);
            return applySelfConditioningRef(embeddings, zero);
        }
    }

    /** Writes encoder projection rows through KVCache2 without adapting through KvBufferCache. */
    public void encodeTextToCacheRef(TensorRef inputIds, KvCacheSession session) {
        ensureTensorRefWeights();
        Preconditions.checkArgument(inputIds.dims() == 2, "inputIds must be [batch, sequence]");
        int batch = inputIds.shape().first();
        int sequence = inputIds.shape().last();
        int hidden = diffusionConfig.textConfig.hiddenSize;
        try (TensorRef embeddings = lighter.allocate(DType.F32, TensorShape.of(batch * sequence, hidden));
             KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
            for (int b = 0; b < batch; b++) {
                for (int position = 0; position < sequence; position++) {
                    int token = (int) inputIds.get(b, position);
                    int row = b * sequence + position;
                    for (int column = 0; column < hidden; column++) {
                        embeddings.set(decoderEmbeddingWeightsRef.get(token, column), row, column);
                    }
                }
            }
            for (int layer = 0; layer < diffusionConfig.textConfig.numHiddenLayers; layer++) {
                int kvLength = encoderKeyWeightsRef[layer].shape().first();
                try (TensorRef keys = lighter.allocate(DType.F32, TensorShape.of(batch * sequence, kvLength));
                     TensorRef values = lighter.allocate(DType.F32, TensorShape.of(batch * sequence, kvLength))) {
                    lighter.dotProductRows(keys, embeddings, encoderKeyWeightsRef[layer], 0, hidden, 0, kvLength, 0);
                    TensorRef valueWeight = encoderValueWeightsRef[layer] == null
                            ? encoderKeyWeightsRef[layer] : encoderValueWeightsRef[layer];
                    lighter.dotProductRows(values, embeddings, valueWeight, 0, hidden, 0, kvLength, 0);
                    for (int b = 0; b < batch; b++) {
                        for (int position = 0; position < sequence; position++) {
                            int row = b * sequence + position;
                            try (TensorRef key = keys.slice(row); TensorRef value = values.slice(row)) {
                                writer.write(layer, row, key, value);
                            }
                        }
                    }
                }
            }
            writer.advanceLength(batch * sequence);
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

        try (TensorRef promptIds = tokenTensorRef(promptTokens);
             KvCacheSession kvSession = newKvCacheSession();
             TensorRef canvas = lighter.allocate(DType.F32, TensorShape.of(1, diffusionConfig.canvasLength));
             TensorRef argmax = lighter.allocate(DType.F32, TensorShape.of(1, 2))) {
            encodeTextToCacheRef(promptIds, kvSession);
            initializeCanvasRef(canvas, new Random(seed));

            try (TensorRef hidden = forwardTextOnlyRef(canvas)) {
                for (int position = 0; position < tokensToReturn; position++) {
                    try (TensorRef logits = logitsForCanvasPositionRef(hidden, 0, position)) {
                        lighter.argMax(new ArgMax(logits).into(argmax).offsetAndLength(0,
                                diffusionConfig.textConfig.vocabSize));
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
        ensureTensorRefWeights();
        return new EmbedInput(this) {
            @Override
            public TensorRef inputTokenToEmbedding(int inputToken, int position) {
                try (TensorRef row = decoderEmbeddingWeightsRef.slice(inputToken)) {
                    TensorRef embedding = makeDenseTensorRef(1, diffusionConfig.textConfig.hiddenSize);
                    try {
                        if (row.dType() == embedding.dType()) {
                            lighter.copy(row, 0, embedding, 0, embedding.shape().last());
                        } else {
                            lighter.reshape(row, embedding);
                        }
                        return embedding;
                    } catch (RuntimeException | Error error) {
                        embedding.close();
                        throw error;
                    }
                }
            }
        };
    }

    @Override
    protected SampleOutputRef loadOutputWeightsRef() {
        ensureTensorRefWeights();
        return new SampleOutputRef() {
            @Override
            public io.teknek.deliverance.generator2.LayerNorm2 outputLayerNorm() {
                throw new UnsupportedOperationException("DiffusionGemma output layer norm is not implemented yet");
            }

            @Override
            public TensorRef outputLogitsWeights() {
                return lmHeadWeightsRef;
            }
        };
    }

    @Override
    protected io.teknek.deliverance.generator2.TransformerBlock2[] loadTransformerBlockWeights2() {
        ensureTensorRefWeights();
        // The decoder attention/denoising block is deliberately not fabricated here. The custom
        // DiffusionGemma skeleton currently stops at self-conditioning.
        return new io.teknek.deliverance.generator2.TransformerBlock2[0];
    }

    /** Required by the shared model contract; V1 execution is intentionally unavailable for this migrated family. */
    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        return new TransformerBlock[0];
    }

    /** Required by the shared model contract; V2 initialization uses {@link #loadOutputWeightsRef()}. */
    @Override
    protected SampleOutput loadOutputWeights() {
        throw new UnsupportedOperationException("DiffusionGemma uses TensorRef output weights");
    }

    @Override
    protected boolean usesTensorRefExecution() {
        return true;
    }

    @Override
    public boolean usesKvCache2Generation() {
        return true;
    }

    /*
     * The decoder canvas is intentionally not routed through the legacy transformer implementation. DiffusionGemma
     * needs cross-attention and denoising masks that are not expressible by the generic TransformerBlock2 contract yet.
     */
    /*
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
                            try (TensorRef scoresRef = TensorRef.borrowed(scores)) {
                                getCompositeOps().scaledSoftMax(new ScaledSoftMax(scale)
                                        .target(scoresRef)
                                        .offsetAndLength(0, sequenceLength));
                            }
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
        for (int i = 0; i < tokenIds.length; i++) tensor.set(tokenIds[i], 0, i);
        return tensor;
    } */

    private TensorRef tokenTensorRef(int[] tokenIds) {
        TensorRef tensor = lighter.allocate(DType.F32, TensorShape.of(1, tokenIds.length));
        try {
            for (int i = 0; i < tokenIds.length; i++) {
                tensor.set(tokenIds[i], 0, i);
            }
            return tensor;
        } catch (RuntimeException | Error error) {
            tensor.close();
            throw error;
        }
    }

    private void initializeCanvasRef(TensorRef canvas, Random random) {
        for (int position = 0; position < diffusionConfig.canvasLength; position++) {
            canvas.set(random.nextInt(diffusionConfig.textConfig.vocabSize), 0, position);
        }
    }

    private String decodeTokens(int[] tokenIds, boolean skipSpecialTokens) {
        return tokenizer.decode(new TokenIds(tokenIds), skipSpecialTokens, false, false, false);
    }

    private static float elapsedMs(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000.0f;
    }

    private void ensureTensorRefWeights() {
        if (selfConditioningRef != null) {
            return;
        }
        decoderEmbeddingWeightsRef = registerModelTensorRef(loadRefFirstPresent("decoder embeddings",
                "model.decoder.embed_tokens.weight", "model.encoder.language_model.embed_tokens.weight"));
        lmHeadWeightsRef = registerModelTensorRef(loadRefFirstPresent("lm head", "lm_head.weight",
                "model.decoder.embed_tokens.weight"));
        selfConditioningPreNormRef = registerModelTensorRef(loadRefFirstPresent("self-conditioning pre norm",
                "model.decoder.self_conditioning.pre_norm.weight"));
        selfConditioningGateRef = registerModelTensorRef(loadRefFirstPresent("self-conditioning gate",
                "model.decoder.self_conditioning.gate_proj.weight"));
        selfConditioningUpRef = registerModelTensorRef(loadRefFirstPresent("self-conditioning up",
                "model.decoder.self_conditioning.up_proj.weight"));
        selfConditioningDownRef = registerModelTensorRef(loadRefFirstPresent("self-conditioning down",
                "model.decoder.self_conditioning.down_proj.weight"));
        encoderKeyWeightsRef = new TensorRef[diffusionConfig.textConfig.numHiddenLayers];
        encoderValueWeightsRef = new TensorRef[diffusionConfig.textConfig.numHiddenLayers];
        for (int layer = 0; layer < encoderKeyWeightsRef.length; layer++) {
            encoderKeyWeightsRef[layer] = registerModelTensorRef(loadRefFirstPresent(
                    "encoder layer " + layer + " key", layerNames(layer, "self_attn.k_proj.weight")));
            encoderValueWeightsRef[layer] = loadRefFirstPresentOptional(layerNames(layer, "self_attn.v_proj.weight"));
            if (encoderValueWeightsRef[layer] != null) {
                encoderValueWeightsRef[layer] = registerModelTensorRef(encoderValueWeightsRef[layer]);
            }
        }
        selfConditioningRef = new DiffusionGemmaSelfConditioning(diffusionConfig.textConfig,
                selfConditioningPreNormRef, selfConditioningGateRef, selfConditioningUpRef, selfConditioningDownRef,
                lighter, compositeOps);
    }

    private String[] layerNames(int layer, String suffix) {
        return new String[] {"model.decoder.layers." + layer + "." + suffix,
                "model.encoder.language_model.layers." + layer + "." + suffix};
    }

    private TensorRef loadRefFirstPresent(String description, String... names) {
        TensorRef ref = loadRefFirstPresentOptional(names);
        if (ref == null) {
            throw new IllegalStateException("DiffusionGemma checkpoint missing " + description + ": "
                    + String.join(", ", names));
        }
        return ref;
    }

    private TensorRef loadRefFirstPresentOptional(String... names) {
        DType qType = modelQType.orElse(modelDType);
        for (String name : names) {
            if (weights.isWeightPresent(name)) {
                return loadAndMaybeQuantizedExcluding1DTensors(name, weights::loadRef, qType);
            }
        }
        return null;
    }

}
