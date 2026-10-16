package io.teknek.deliverance.model.embeddinggemma2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.embedding.PoolingLayer;
import io.teknek.deliverance.embedding.PoolingType;
import io.teknek.deliverance.generator.EmbedInput;
import io.teknek.deliverance.generator.SampleOutput;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.grace.EncodeOptions;
import io.teknek.deliverance.grace.Encoding;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import io.teknek.deliverance.tensor.TensorShape;

import java.util.Arrays;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.nio.file.Path;

/** Text-only EmbeddingGemma 2 encoder exposed through Deliverance's embedding API. */
public final class EmbeddingGemma2Model extends AbstractModel {
    private final EmbeddingGemma2TextConfig textConfig;
    private TensorRef embeddingTable;
    private TensorRef pleProjection;
    private TensorRef pleNorm;
    private TensorRef finalNorm;
    private TensorRef embeddingProjection;
    private LayerWeights[] layers;

    /** Creates an EmbeddingGemma 2 model using the standard reflective model constructor. */
    public EmbeddingGemma2Model(InferenceType inferenceType, Config config, WeightLoader weights,
            PreTrainedTokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType,
            ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry,
            TensorAllocator tensorAllocator, KvBufferCacheSettings kvBufferCacheSettings,
            ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType,
                configurableTensorProvider, metricRegistry, tensorAllocator, kvBufferCacheSettings, toolCallParser,
                pool, tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
        this.textConfig = ((EmbeddingGemma2Config) config).textConfig;
    }

    @Override
    protected boolean usesTensorRefExecution() {
        return true;
    }

    @Override
    protected EmbedInput loadInputWeights() {
        embeddingTable = loadRef("embed_tokens.weight");
        pleProjection = loadRef("ple.per_layer_model_projection.weight");
        pleNorm = loadNormRef("ple.per_layer_projection_norm.weight");
        finalNorm = loadNormRef("norm.weight");
        embeddingProjection = loadRef("embedding_projection.weight");
        layers = loadLayers();
        return new EmbedInput(this) {
            @Override
            public TensorRef inputTokenToEmbedding(int inputToken, int position) {
                TensorRef output = makeF32(1, textConfig.hiddenSize);
                new EmbeddingGemma2ScaledWordEmbeddingOperation().apply(output, embeddingTable,
                        new int[] {inputToken}, roundedEmbeddingScale(), lighter);
                return output;
            }
        };
    }

    @Override
    protected io.teknek.deliverance.generator2.TransformerBlock2[] loadTransformerBlockWeights2() {
        // The encoder has bidirectional attention and does not use the causal TransformerBlock2 path.
        return new io.teknek.deliverance.generator2.TransformerBlock2[0];
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        throw new UnsupportedOperationException("EmbeddingGemma 2 has no language-model output head");
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        return new TransformerBlock[0];
    }

    @Override
    protected float[] timedEmbedding(String input, PoolingType poolingType) {
        return embedText(input, EmbeddingGemma2EmbeddingOptions.defaults());
    }

    /** Encodes text with task prompting, optional MRL truncation, and post-truncation normalization. */
    public float[] embed(String input, EmbeddingGemma2EmbeddingOptions options) {
        return embedText(input, options);
    }

    private float[] embedText(String input, EmbeddingGemma2EmbeddingOptions options) {
        if (input == null || input.isEmpty()) {
            throw new IllegalArgumentException("Text input must not be empty");
        }
        String prompted = applyPrompt(input, options);
        Encoding encoding = tokenizer.encode(prompted, EncodeOptions.defaults());
        int[] tokenIds = encoding.inputIds();
        if (tokenIds.length == 0 || tokenIds.length > textConfig.maxPositionEmbeddings) {
            throw new IllegalArgumentException("Text input must contain between one and max-position tokens");
        }
        int[] attentionMask = encoding.attentionMask();
        try (TensorRef hidden = encode(tokenIds, attentionMask);
             TensorRef projected = poolAndProject(hidden, attentionMask)) {
            int outputDimension = options.truncateDimension() == null
                    ? textConfig.embeddingDim : options.truncateDimension();
            if (outputDimension > textConfig.embeddingDim) {
                throw new IllegalArgumentException("Requested dimension exceeds model embedding dimension "
                        + textConfig.embeddingDim);
            }
            float[] result = new float[outputDimension];
            for (int column = 0; column < result.length; column++) {
                result[column] = projected.get(0, column);
            }
            if (options.normalizeEmbeddings()) {
                normalize(result);
                if (usesBf16Reference()) {
                    for (int column = 0; column < result.length; column++) {
                        result[column] = roundFloatToBf16(result[column]);
                    }
                }
            }
            return result;
        }
    }

    private String applyPrompt(String input, EmbeddingGemma2EmbeddingOptions options) {
        if (options.promptOverride() != null) {
            return options.promptOverride() + input;
        }
        if (options.taskPrompt() == null) {
            return input;
        }
        Optional<Path> root = weights.modelRoot();
        if (root.isEmpty()) {
            throw new IllegalArgumentException("Task prompt requires model metadata or an explicit prompt override");
        }
        try {
            Path metadata = root.get().resolve("config_sentence_transformers.json");
            if (!java.nio.file.Files.isRegularFile(metadata)) {
                throw new IllegalArgumentException("No Sentence Transformers prompt metadata found for task "
                        + options.taskPrompt());
            }
            var prompts = JsonUtils.om.readTree(metadata.toFile()).path("prompts");
            String prompt = prompts.path(options.taskPrompt()).asText(null);
            if (prompt == null) {
                throw new IllegalArgumentException("Unknown EmbeddingGemma 2 task prompt: " + options.taskPrompt());
            }
            return prompt + input;
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException("Unable to read task prompt metadata", e);
        }
    }

    private TensorRef encode(int[] tokenIds, int[] attentionMask) {
        TensorRef hidden = makeF32(tokenIds.length, textConfig.hiddenSize);
        try (var ignored = InferenceProfiler.timer(metricRegistry, "embeddinggemma2.embedding").time()) {
            new EmbeddingGemma2ScaledWordEmbeddingOperation().apply(hidden, embeddingTable, tokenIds,
                    roundedEmbeddingScale(), lighter);
        }
        roundBoundary(hidden);
        TensorRef perLayer = makeF32(tokenIds.length,
                textConfig.numHiddenLayers * textConfig.hiddenSizePerLayerInput);
        try (var ignored = InferenceProfiler.timer(metricRegistry, "embeddinggemma2.ple_projection").time()) {
            new EmbeddingGemma2PleProjectionOperation(textConfig.hiddenSize, textConfig.numHiddenLayers,
                    textConfig.hiddenSizePerLayerInput).apply(perLayer, hidden, pleProjection, lighter, this);
            roundBoundary(perLayer);
            new EmbeddingGemma2PleGroupedRmsNormOperation(textConfig.numHiddenLayers,
                    textConfig.hiddenSizePerLayerInput, textConfig.rmsNormEps).apply(perLayer, pleNorm, lighter);
            roundBoundary(perLayer);
        }
        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            LayerWeights layer = layers[layerIndex];
            try (TensorRef perLayerInput = layerInput(perLayer, layerIndex)) {
                TensorRef next = makeF32(tokenIds.length, textConfig.hiddenSize);
                try (var ignored = InferenceProfiler.timer(metricRegistry,
                        "embeddinggemma2.layer." + layerIndex).time()) {
                    layer.operation.apply(next, hidden, perLayerInput, layer.inputNorm, layer.query, layer.key,
                            layer.value, layer.output, layer.queryNorm, layer.keyNorm, layer.postAttentionNorm,
                            layer.preFeedForwardNorm, layer.gate, layer.up, layer.down, layer.postFeedForwardNorm,
                            layer.pleGate, layer.pleProjection, layer.pleNorm, layer.layerScalar, attentionMask, lighter,
                            this);
                }
                hidden.close();
                hidden = next;
            }
        }
        try (var ignored = InferenceProfiler.timer(metricRegistry, "embeddinggemma2.final_norm").time();
             TensorRef normalized = makeF32(tokenIds.length, textConfig.hiddenSize)) {
             new EmbeddingGemma2RmsNormOperation(textConfig.rmsNormEps).apply(normalized, hidden, finalNorm, lighter);
             lighter.copyStorage(normalized, hidden);
        }
        perLayer.close();
        return hidden;
    }

    private TensorRef poolAndProject(TensorRef hidden, int[] attentionMask) {
        try (var ignored = InferenceProfiler.timer(metricRegistry, "embeddinggemma2.pool_project").time()) {
            return poolAndProjectBody(hidden, attentionMask);
        }
    }

    private TensorRef poolAndProjectBody(TensorRef hidden, int[] attentionMask) {
        TensorRef pooled = makeF32(1, textConfig.hiddenSize);
        int valid = 0;
        for (int row = 0; row < hidden.shape().first(); row++) {
            if (attentionMask == null || attentionMask[row] != 0) {
                valid++;
                for (int column = 0; column < textConfig.hiddenSize; column++) {
                    pooled.set(pooled.get(0, column) + hidden.get(row, column), 0, column);
                }
            }
        }
        if (valid == 0) {
            pooled.close();
            throw new IllegalArgumentException("Attention mask contains no valid tokens");
        }
        lighter.multiplyAll(1.0f / valid, pooled);
        TensorRef projected = makeF32(1, textConfig.embeddingDim);
        io.teknek.deliverance.tensor2.GemmDit.dotProductRows(lighter, this, "embeddinggemma2.output_projection",
                projected, pooled, embeddingProjection, 0, 0, textConfig.hiddenSize, 0,
                textConfig.embeddingDim, 0);
        roundBoundary(projected);
        pooled.close();
        return projected;
    }

    private void roundBoundary(TensorRef target) {
        if (!usesBf16Reference()) return;
        try (TensorRef bf16 = lighter.reshape(target, DType.BF16,
                io.teknek.deliverance.tensor2.Threading.PROVIDER)) {
            lighter.reshape(bf16, target, io.teknek.deliverance.tensor2.Threading.PROVIDER);
        }
    }

    private LayerWeights[] loadLayers() {
        LayerWeights[] result = new LayerWeights[textConfig.numHiddenLayers];
        DType qType = modelQType.orElse(modelDType);
        for (int index = 0; index < result.length; index++) {
            String base = "layers." + index + ".";
            String layerType = textConfig.layerTypes.get(index);
            int headWidth = textConfig.headDimFor(index);
            int keyValueHeads = textConfig.keyValueHeadsFor(index);
            double ropeTheta = ropeTheta(layerType);
            result[index] = new LayerWeights(
                    new EmbeddingGemma2EncoderLayerOperation(textConfig.hiddenSize, textConfig.intermediateSize,
                            textConfig.numAttentionHeads, keyValueHeads, headWidth,
                            textConfig.hiddenSizePerLayerInput, textConfig.rmsNormEps, layerType,
                            textConfig.slidingWindow, ropeTheta, usesBf16Reference(),
                            textConfig.maxPositionEmbeddings, getPool().getUnderlying()),
                    loadNormRef(base + "input_layernorm.weight", qType),
                    loadRef(base + "self_attn.q_proj.weight", qType),
                    loadRef(base + "self_attn.k_proj.weight", qType),
                    loadRef(base + "self_attn.v_proj.weight", qType),
                    loadRef(base + "self_attn.o_proj.weight", qType),
                    loadNormRef(base + "self_attn.q_norm.weight", qType),
                    loadNormRef(base + "self_attn.k_norm.weight", qType),
                    loadNormRef(base + "post_attention_layernorm.weight", qType),
                    loadNormRef(base + "pre_feedforward_layernorm.weight", qType),
                    loadRef(base + "mlp.gate_proj.weight", qType),
                    loadRef(base + "mlp.up_proj.weight", qType),
                    loadRef(base + "mlp.down_proj.weight", qType),
                    loadNormRef(base + "post_feedforward_layernorm.weight", qType),
                    loadRef(base + "ple_block.per_layer_input_gate.weight", qType),
                    loadRef(base + "ple_block.per_layer_projection.weight", qType),
                    loadNormRef(base + "ple_block.post_per_layer_input_norm.weight", qType),
                    loadScalarOrDefault(base + "layer_scalar", 1.0f));
        }
        return result;
    }

    private TensorRef layerInput(TensorRef perLayer, int layer) {
        TensorRef result = makeF32(perLayer.shape().first(), textConfig.hiddenSizePerLayerInput);
        int offset = layer * textConfig.hiddenSizePerLayerInput;
        for (int row = 0; row < result.shape().first(); row++) {
            for (int column = 0; column < result.shape().last(); column++) {
                result.set(perLayer.get(row, offset + column), row, column);
            }
        }
        return result;
    }

    private TensorRef loadRef(String name) {
        return loadRef(name, modelQType.orElse(modelDType));
    }

    private TensorRef loadRef(String name, DType qType) {
        return registerModelTensorRef(loadRawRef(name, qType));
    }

    private TensorRef loadNormRef(String name) {
        return loadNormRef(name, modelQType.orElse(modelDType));
    }

    private TensorRef loadNormRef(String name, DType qType) {
        TensorRef source = loadRawRef(name, qType);
        if (source.dType() == DType.F32) {
            return registerModelTensorRef(source);
        }
        TensorRef converted = lighter.reshape(source, DType.F32,
                io.teknek.deliverance.tensor2.Threading.PROVIDER);
        source.close();
        return registerModelTensorRef(converted);
    }

    private TensorRef loadRawRef(String name, DType qType) {
        String[] candidates = {"language_model." + name, name};
        for (String candidate : candidates) {
            if (weights.isWeightPresent(candidate)) {
                return loadAndMaybeQuantizedExcluding1DTensors(candidate, weights::loadRef, qType);
            }
        }
        throw new NoSuchElementException(Arrays.toString(candidates) + " not found in weights "
                + weights.tensorInfoMap());
    }

    private float loadScalarOrDefault(String name, float defaultValue) {
        String[] candidates = {"language_model." + name, name};
        for (String candidate : candidates) {
            if (weights.isWeightPresent(candidate)) {
                try (TensorRef scalar = weights.loadRef(candidate)) {
                    return scalar.get(0, 0);
                }
            }
        }
        return defaultValue;
    }

    private double ropeTheta(String layerType) {
        Map<String, Object> parameters = textConfig.ropeParameters.get(layerType);
        Object theta = parameters == null ? null : parameters.get("rope_theta");
        if (!(theta instanceof Number number)) {
            throw new IllegalArgumentException("Missing rope_theta for " + layerType);
        }
        return number.doubleValue();
    }

    private TensorRef makeF32(int... shape) {
        return lighter.allocate(DType.F32, TensorShape.of(shape));
    }

    private float roundedEmbeddingScale() {
        // The production checkpoint is BF16. F32 tiny checkpoints use the exact mathematical scale.
        return usesBf16Reference() ? 22.625f : (float) Math.sqrt(textConfig.hiddenSize);
    }

    private boolean usesBf16Reference() {
        return embeddingTable != null && embeddingTable.dType() == DType.BF16;
    }

    private void normalize(float[] values) {
        double sum = 0.0;
        for (float value : values) sum += (double) value * value;
        float inverse = (float) (1.0 / Math.sqrt(sum));
        for (int i = 0; i < values.length; i++) values[i] *= inverse;
    }

    private static float roundFloatToBf16(float value) {
        int bits = Float.floatToIntBits(value);
        int rounding = 0x7fff + ((bits >>> 16) & 1);
        return Float.intBitsToFloat((bits + rounding) & 0xffff0000);
    }

    private record LayerWeights(EmbeddingGemma2EncoderLayerOperation operation, TensorRef inputNorm,
            TensorRef query, TensorRef key, TensorRef value, TensorRef output, TensorRef queryNorm,
            TensorRef keyNorm, TensorRef postAttentionNorm, TensorRef preFeedForwardNorm, TensorRef gate,
            TensorRef up, TensorRef down, TensorRef postFeedForwardNorm, TensorRef pleGate,
            TensorRef pleProjection, TensorRef pleNorm, float layerScalar) {
    }
}
