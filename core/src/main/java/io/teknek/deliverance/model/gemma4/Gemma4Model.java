package io.teknek.deliverance.model.gemma4;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.math.FloatConversions;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.TokenRenderer;
import io.teknek.deliverance.model.llama.LlamaModel;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.*;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tokenizer.Tokenizer;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;

import static io.teknek.deliverance.tensor.AbstractTensorUtils.quantize;

public class Gemma4Model extends LlamaModel {
    private static final Logger logger = LoggerFactory.getLogger(Gemma4Model.class);
    private static final boolean DEBUG_LAYER_SUMMARIES = false;
    private final ThreadLocal<Map<String, SharedKeyValues>> sharedKeyValues = new ThreadLocal<>();
    private volatile AbstractTensor perLayerModelProjectionWeights;
    private volatile AbstractTensor perLayerProjectionNormWeights;
    private volatile AbstractTensor embedTokenWeights;

    public record SharedKeyValues(AbstractTensor key, AbstractTensor value) implements AutoCloseable {
        @Override
        public void close() {
            key.close();
            value.close();
        }
    }

    public Gemma4Model(
            InferenceType inferenceType,
            Config config,
            WeightLoader weights,
            Tokenizer tokenizer,
            DType workingDType,
            DType workingQType,
            Optional<DType> modelQType,
            ConfigurableTensorProvider configurableTensorProvider,
            MetricRegistry metricRegistry,
            TensorAllocator arrayQueueTensorAllocator,
            KvBufferCacheSettings kvBufferCacheSettings,
            TokenRenderer tokenRenderer,
            ToolCallParser toolCallParser,
            WrappedForkJoinPool pool
    ) {
        super(
                inferenceType,
                config,
                weights,
                tokenizer,
                workingDType,
                workingQType,
                modelQType,
                configurableTensorProvider,
                metricRegistry,
                arrayQueueTensorAllocator,
                kvBufferCacheSettings,
                tokenRenderer,
                toolCallParser,
                pool
        );
    }

    @Override
    protected EmbedInput loadInputWeights() {
        if (embedTokenWeights == null) {
            embedTokenWeights = quantize(weights.load(resolveTextModelWeight("embed_tokens.weight")), workingDType);
        }
        final float embeddingScalingFactor = FloatConversions.bFloat16ToFloat32(
                FloatConversions.float32ToBFloat16((float) Math.pow(config.embeddingLength, 0.5))
        );
        return new EmbedInput(this) {
            @Override
            public AbstractTensor inputTokenToEmbedding(int inputToken, int position) {
                AbstractTensor embedding = makeDenseTensor(config.embeddingLength);
                try (AbstractTensor at = embedTokenWeights.slice(true, inputToken)) {
                    AbstractTensor source = at;
                    if (embedTokenWeights.dType() != embedding.dType()) {
                        source = configurableTensorProvider.get().quantize(at, embedding.dType(), 0, config.embeddingLength);
                    }
                    embedding.copyFrom(source, 0, 0, config.embeddingLength);
                    if (source != at) {
                        source.close();
                    }
                }
                configurableTensorProvider.get().scale(embeddingScalingFactor, embedding, 0, config.embeddingLength);
                return embedding;
            }
        };
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        final LayerNorm outputLayerNorm = weights.isWeightPresent(resolveTextModelRoot() + "norm.weight")
                ? new RmsNorm(this, quantize(weights.load(resolveTextModelWeight("norm.weight")), qType), 0.0f, metricRegistry)
                : new IdentityLayerNorm(this, metricRegistry);
        AbstractTensor classificationWeights = weights.isWeightPresent(resolveTextOutputWeight("lm_head.weight"))
                ? quantize(weights.load(resolveTextOutputWeight("lm_head.weight")), workingDType)
                : embedTokenWeights == null ? embedTokenWeights = quantize(weights.load(resolveTextModelWeight("embed_tokens.weight")), workingDType)
                : embedTokenWeights;
        configurableTensorProvider.get().registerModelTensor(classificationWeights);
        return new SampleOutput() {
            @Override
            public LayerNorm getOutputLayerNorm() {
                return outputLayerNorm;
            }

            @Override
            public AbstractTensor getOutputLogitsWeights() {
                return classificationWeights;
            }
        };
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        Gemma4Config gemma4Config = gemma4Config();
        DType qType = modelQType.orElse(this.modelDType);
        TransformerBlock[] blocks = new TransformerBlock[config.dctx().numberOfLayers];
        String root = resolveTextModelRoot();
        for (int i = config.dctx().layerStart; i < config.dctx().layerEnd; i++) {
            int relativeLayer = i - config.dctx().layerStart;
            String base = root + "layers." + i + ".";
            String layerType = gemma4Config.layerTypes.get(i);
            int sharedSource = gemma4Config.getSharedKvSourceLayer(i);
            boolean kvSharedLayer = sharedSource >= 0;

            Gemma4CausalSelfAttention attention = new Gemma4CausalSelfAttention(
                    this,
                    i,
                    layerType,
                    kvSharedLayer,
                    gemma4Config.storesSharedKvState(i),
                    quantize(weights.load(base + "self_attn.q_proj.weight"), qType),
                    quantize(weights.load(base + "self_attn.q_norm.weight"), qType),
                    kvSharedLayer ? Optional.empty() : Optional.of(quantize(weights.load(base + "self_attn.k_proj.weight"), qType)),
                    kvSharedLayer ? Optional.empty() : Optional.of(quantize(weights.load(base + "self_attn.v_proj.weight"), qType)),
                    kvSharedLayer ? Optional.empty() : Optional.of(quantize(weights.load(base + "self_attn.k_norm.weight"), qType)),
                    quantize(weights.load(base + "self_attn.o_proj.weight"), qType),
                    configurableTensorProvider,
                    metricRegistry
            );

            FeedForward mlp = new VariableMLPBlock(
                    this,
                    gemma4Config.activationFunction,
                    quantize(weights.load(base + "mlp.gate_proj.weight"), qType),
                    quantize(weights.load(base + "mlp.down_proj.weight"), qType),
                    quantize(weights.load(base + "mlp.up_proj.weight"), qType),
                    gemma4Config.getLayerHiddenLength(i),
                    configurableTensorProvider
            );

            Optional<AbstractTensor> perLayerInputGate = Optional.empty();
            Optional<AbstractTensor> perLayerProjection = Optional.empty();
            Optional<LayerNorm> postPerLayerInputNorm = Optional.empty();
            if (gemma4Config.hiddenSizePerLayerInput != null && weights.isWeightPresent(base + "per_layer_input_gate.weight")) {
                perLayerInputGate = Optional.of(quantize(weights.load(base + "per_layer_input_gate.weight"), qType));
                perLayerProjection = Optional.of(quantize(weights.load(base + "per_layer_projection.weight"), qType));
                    postPerLayerInputNorm = Optional.of(new RmsNorm(this,
                        quantize(weights.load(base + "post_per_layer_input_norm.weight"), qType), 0.0f, metricRegistry));
            }

            blocks[relativeLayer] = new Gemma4TransformerBlock(
                    this,
                    i,
                    new RmsNorm(this, quantize(weights.load(base + "input_layernorm.weight"), qType), 0.0f, metricRegistry),
                    attention,
                    new RmsNorm(this, quantize(weights.load(base + "post_attention_layernorm.weight"), qType), 0.0f, metricRegistry),
                    new RmsNorm(this, quantize(weights.load(base + "pre_feedforward_layernorm.weight"), qType), 0.0f, metricRegistry),
                    mlp,
                    new RmsNorm(this, quantize(weights.load(base + "post_feedforward_layernorm.weight"), qType), 0.0f, metricRegistry),
                    configurableTensorProvider,
                    perLayerInputGate,
                    perLayerProjection,
                    postPerLayerInputNorm,
                    gemma4Config.hiddenSizePerLayerInput == null ? 0 : gemma4Config.hiddenSizePerLayerInput,
                    loadScalar(base + "layer_scalar"),
                    gemma4Config.activationFunction
            );
        }
        return blocks;
    }

    @Override
    public AbstractTensor batchForward(int[] tokenIds, int startPos, KvBufferCache.KvBuffer kvbuf,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        metricRegistry.histogram("gemma4.batch.tokens").update(tokenIds.length);
        metricRegistry.histogram("gemma4.batch.start_pos").update(startPos);
        return withSharedKeyValues(() -> {
            Timer.Context embedTimer = metricRegistry.timer("gemma4.batch_forward.embed").time();
            AbstractTensor embedding;
            try {
                embedding = embedInput.batchInputsToEmbeddings(tokenIds, startPos);
            } finally {
                embedTimer.stop();
            }

            Timer.Context pleTimer = metricRegistry.timer("gemma4.batch_forward.ple").time();
            AbstractTensor[] perLayerInputs;
            try {
                perLayerInputs = computePerLayerInputs(tokenIds, embedding);
            } finally {
                pleTimer.stop();
            }

            Timer.Context forwardTimer = metricRegistry.timer("gemma4.batch_forward.forward").time();
            try {
                return forwardGemma4(embedding, perLayerInputs, startPos, kvbuf, tensorReducer);
            } finally {
                forwardTimer.stop();
            }
        });
    }

    @Override
    public AbstractTensor batchForward(int[] tokenIds, int startPos, KvBufferCache.KvBuffer kvbuf) {
        return batchForward(tokenIds, startPos, kvbuf, Optional.empty());
    }

    @Override
    public AbstractTensor forward(int tokenId, int pos, KvBufferCache.KvBuffer kvbuf,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        return withSharedKeyValues(() -> {
            AbstractTensor embedding = embedInput.inputTokenToEmbedding(tokenId, pos);
            AbstractTensor[] perLayerInputs = computePerLayerInputs(new int[]{tokenId}, embedding);
            return forwardGemma4(embedding, perLayerInputs, pos, kvbuf, tensorReducer);
        });
    }

    @Override
    public AbstractTensor forward(AbstractTensor embedding, int startPos, KvBufferCache.KvBuffer kvbuf,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        return withSharedKeyValues(() -> forwardGemma4(embedding, null, startPos, kvbuf, tensorReducer));
    }

    @Override
    protected Response postProcessResponse(Response response) {
        Gemma4ResponseParser.Parsed parsed = Gemma4ResponseParser.parse(
                response.responseTextWithSpecialTokens,
                response.responseText
        );
        return response.copyWithText(parsed.content(), response.responseTextWithSpecialTokens, parsed.reasoning());
    }

    public SharedKeyValues getSharedKeyValues(String layerType) {
        Map<String, SharedKeyValues> current = sharedKeyValues.get();
        if (current == null || !current.containsKey(layerType)) {
            throw new IllegalStateException("Missing shared kv state for layer type " + layerType);
        }
        return current.get(layerType);
    }

    public void putSharedKeyValues(String layerType, AbstractTensor key, AbstractTensor value) {
        Map<String, SharedKeyValues> current = sharedKeyValues.get();
        if (current == null) {
            throw new IllegalStateException("Shared kv state not initialized");
        }
        SharedKeyValues previous = current.remove(layerType);
        if (previous != null) {
            previous.close();
        }
        AbstractTensor keyCopy = makeDenseTensor(TensorShape.of(key.shape().first(), key.shape().last()));
        AbstractTensor valueCopy = makeDenseTensor(TensorShape.of(value.shape().first(), value.shape().last()));
        keyCopy.copyFrom(key, 0, 0, (int) key.size());
        valueCopy.copyFrom(value, 0, 0, (int) value.size());
        current.put(layerType, new SharedKeyValues(keyCopy, valueCopy));
    }

    private AbstractTensor forwardGemma4(AbstractTensor embedding, AbstractTensor[] perLayerInputs, int startPos,
            KvBufferCache.KvBuffer kvbuf, Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        for (int i = config.dctx().layerStart; i < config.dctx().layerEnd; i++) {
            int relativeLayer = i - config.dctx().layerStart;
            AbstractTensor ref = embedding;
            AbstractTensor perLayerInput = perLayerInputs == null ? null : perLayerInputs[i];
            Timer.Context layerTimer = metricRegistry.timer("gemma4.layer." + i + "." + gemma4Config().layerTypes.get(i)).time();
            try {
                embedding = ((Gemma4TransformerBlock) transformerBlocks[relativeLayer]).forward(embedding, perLayerInput,
                        startPos, kvbuf, tensorReducer);
            } finally {
                layerTimer.stop();
            }
            if (DEBUG_LAYER_SUMMARIES) {
                logLayerSummary(i, gemma4Config().layerTypes.get(i), embedding);
            }
            ref.close();
        }
        if (perLayerInputs != null) {
            for (AbstractTensor perLayerInput : perLayerInputs) {
                if (perLayerInput != null) {
                    perLayerInput.close();
                }
            }
        }
        return embedding;
    }

    /**
     * Emits a compact summary of row 0 of the current hidden state. This is intended for locating
     * the first layer where prefill activations become suspicious without dumping full tensors.
     */
    void logLayerSummary(int layerIndex, String layerType, AbstractTensor hiddenStates) {
        if (hiddenStates.shape().first() < 1 || hiddenStates.shape().last() < 1) {
            logger.info("gemma4 layer_summary layer={} type={} empty=true", layerIndex, layerType);
            return;
        }
        int width = hiddenStates.shape().last();
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        double sum = 0.0d;
        double sumSquares = 0.0d;
        StringBuilder first = new StringBuilder();
        int preview = Math.min(8, width);
        for (int i = 0; i < width; i++) {
            float v = hiddenStates.get(0, i);
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
            sum += v;
            sumSquares += (double) v * v;
            if (i < preview) {
                if (i > 0) {
                    first.append(',');
                }
                first.append(String.format(java.util.Locale.ROOT, "%.4f", v));
            }
        }
        double mean = sum / width;
        double l2 = Math.sqrt(sumSquares);
        logger.info("gemma4 layer_summary layer={} type={} row0_min={} row0_max={} row0_mean={} row0_l2={} row0_first8=[{}]",
                layerIndex,
                layerType,
                String.format(java.util.Locale.ROOT, "%.6f", min),
                String.format(java.util.Locale.ROOT, "%.6f", max),
                String.format(java.util.Locale.ROOT, "%.6f", mean),
                String.format(java.util.Locale.ROOT, "%.6f", l2),
                first);
    }

    private synchronized void ensurePerLayerWeightsLoaded() {
        Gemma4Config gemma4Config = gemma4Config();
        if (gemma4Config.hiddenSizePerLayerInput == null || perLayerModelProjectionWeights != null) {
            return;
        }
        DType qType = modelQType.orElse(this.modelDType);
        perLayerModelProjectionWeights = quantize(weights.load(resolveTextModelWeight("per_layer_model_projection.weight")), qType);
        perLayerProjectionNormWeights = quantize(weights.load(resolveTextModelWeight("per_layer_projection_norm.weight")), qType);
    }

    private AbstractTensor[] computePerLayerInputs(int[] tokenIds, AbstractTensor embeddings) {
        Gemma4Config gemma4Config = gemma4Config();
        if (gemma4Config.hiddenSizePerLayerInput == null) {
            return null;
        }
        ensurePerLayerWeightsLoaded();
        int batchSize = tokenIds.length;
        int packedLength = gemma4Config.numberOfLayers * gemma4Config.hiddenSizePerLayerInput;
        try (
                AbstractTensor tokenIdentity = makeDenseTensor(batchSize, packedLength);
                AbstractTensor projected = makeDenseTensor(batchSize, packedLength)
        ) {
            computeTokenIdentityPerLayerInputs(tokenIds, tokenIdentity, packedLength);
            computeProjectedPerLayerInputs(embeddings, projected, packedLength, gemma4Config);
            combinePerLayerInputs(projected, tokenIdentity, packedLength);
            return splitPerLayerInputs(projected, batchSize, gemma4Config.numberOfLayers, gemma4Config.hiddenSizePerLayerInput);
        }
    }

    private float loadScalar(String name) {
        if (!weights.isWeightPresent(name)) {
            return 1.0f;
        }
        try (AbstractTensor tensor = weights.load(name)) {
            return tensor.shape().dims() == 1 ? tensor.get(0) : tensor.get(0, 0);
        }
    }

    private <T> T withSharedKeyValues(java.util.function.Supplier<T> supplier) {
        Map<String, SharedKeyValues> previous = sharedKeyValues.get();
        Map<String, SharedKeyValues> current = new HashMap<>();
        sharedKeyValues.set(current);
        try {
            return supplier.get();
        } finally {
            current.values().forEach(SharedKeyValues::close);
            if (previous == null) {
                sharedKeyValues.remove();
            } else {
                sharedKeyValues.set(previous);
            }
        }
    }

    private Gemma4Config gemma4Config() {
        return (Gemma4Config) config;
    }

    private float perLayerEmbeddingScale() {
        Gemma4Config cfg = gemma4Config();
        return cfg.hiddenSizePerLayerInput == null ? 1.0f : (float) Math.sqrt(cfg.hiddenSizePerLayerInput);
    }

    private float perLayerInputScale() {
        return (float) Math.pow(2.0, -0.5);
    }

    private float perLayerModelProjectionScale() {
        return (float) Math.pow(config.embeddingLength, -0.5);
    }

    /**
     * Populates the packed token-identity component of Gemma4 PLE from the per-layer embedding table,
     * then applies the upstream `sqrt(hidden_size_per_layer_input)` scale.
     */
    void computeTokenIdentityPerLayerInputs(int[] tokenIds, AbstractTensor tokenIdentity, int packedLength) {
        for (int b = 0; b < tokenIds.length; b++) {
            copyPerLayerEmbeddingRow(tokenIds[b], tokenIdentity, b, packedLength);
        }
        configurableTensorProvider.get().scale(perLayerEmbeddingScale(), tokenIdentity, 0, packedLength);
    }

    /**
     * Computes the context-dependent PLE projection, applies the upstream projection scale, and then
     * normalizes each per-layer slice with `per_layer_projection_norm`.
     */
    void computeProjectedPerLayerInputs(AbstractTensor embeddings, AbstractTensor projected, int packedLength, Gemma4Config gemma4Config) {
        configurableTensorProvider.get().dotProductChunk(projected, embeddings, perLayerModelProjectionWeights, 0,
                gemma4Config.embeddingLength, 0, packedLength);
        configurableTensorProvider.get().scale(perLayerModelProjectionScale(), projected, 0, packedLength);
        Gemma4RmsNormSupport.applyInPlace(projected, gemma4Config.numberOfLayers,
                gemma4Config.hiddenSizePerLayerInput, gemma4Config.layerNormEps, perLayerProjectionNormWeights);
    }

    /**
     * Upstream combines token-identity and projected PLE components as `(identity + projected) / sqrt(2)`.
     */
    void combinePerLayerInputs(AbstractTensor projected, AbstractTensor tokenIdentity, int packedLength) {
        Gemma4PleSupport.combinePerLayerInputs(configurableTensorProvider, projected, tokenIdentity, perLayerInputScale(), packedLength);
    }

    AbstractTensor[] splitPerLayerInputs(AbstractTensor projected, int batchSize, int numberOfLayers, int hiddenSizePerLayerInput) {
        return Gemma4PleSupport.splitPerLayerInputs(projected, batchSize, numberOfLayers, hiddenSizePerLayerInput,
                this::makeDenseTensor);
    }

    private String resolveTextModelRoot() {
        List<String> candidates = List.of("model.language_model.", "language_model.model.", "language_model.", "model.");
        for (String candidate : candidates) {
            if (weights.isWeightPresent(candidate + "embed_tokens.weight")
                    || weights.isWeightPresent(candidate + "layers.0.self_attn.q_proj.weight")) {
                return candidate;
            }
        }
        throw new IllegalArgumentException("Unable to locate Gemma4 text model root");
    }

    private String resolveTextModelWeight(String suffix) {
        String root = resolveTextModelRoot();
        String name = root + suffix;
        if (weights.isWeightPresent(name) || weights.isWeightPresent(name + "-part-0")) {
            return name;
        }
        throw new IllegalArgumentException("Missing Gemma4 text weight " + name);
    }

    private String resolveTextOutputWeight(String suffix) {
        List<String> candidates = List.of(suffix, "language_model." + suffix, resolveTextModelRoot() + suffix);
        for (String candidate : candidates) {
            if (weights.isWeightPresent(candidate)) {
                return candidate;
            }
        }
        return suffix;
    }

    private void copyPerLayerEmbeddingRow(int tokenId, AbstractTensor destination, int batchIndex, int packedLength) {
        try (AbstractTensor row = weights.loadRows(resolveTextModelWeight("embed_tokens_per_layer.weight"), tokenId, 1)) {
            AbstractTensor source = row;
            if (row.dType() != destination.dType()) {
                source = configurableTensorProvider.get().quantize(row, destination.dType(), 0, packedLength);
            }
            destination.copyFrom(source, 0, destination.getOffset(batchIndex, 0), packedLength);
            if (source != row) {
                source.close();
            }
        }
    }
}
