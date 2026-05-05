package io.teknek.deliverance.model.gemma4;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.TokenRenderer;
import io.teknek.deliverance.model.gemma3.Gemma3Model;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tokenizer.Tokenizer;
import io.teknek.deliverance.toolcallparser.ToolCallParser;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;

import static io.teknek.deliverance.tensor.AbstractTensorUtils.quantize;

public class Gemma4Model extends Gemma3Model {
    private final Gemma4Config gemma4Config;
    private final ThreadLocal<Map<Integer, SharedKeyValues>> sharedKeyValues = new ThreadLocal<>();
    private volatile AbstractTensor perLayerEmbeddingTable;
    private volatile AbstractTensor perLayerModelProjectionWeights;
    private volatile AbstractTensor perLayerProjectionNormWeights;
    private final float perLayerEmbeddingScale;
    private final float perLayerInputScale;
    private final float perLayerModelProjectionScale;

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
            ArrayQueueTensorAllocator arrayQueueTensorAllocator,
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
        this.gemma4Config = (Gemma4Config) config;
        this.perLayerEmbeddingScale = gemma4Config.hiddenSizePerLayerInput == null ? 1.0f : (float) Math.sqrt(gemma4Config.hiddenSizePerLayerInput);
        this.perLayerInputScale = (float) Math.pow(2.0, -0.5);
        this.perLayerModelProjectionScale = (float) Math.pow(gemma4Config.embeddingLength, -0.5);
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        TransformerBlock[] blocks = new TransformerBlock[config.dctx().numberOfLayers];
        for (int i = config.dctx().layerStart; i < config.dctx().layerEnd; i++) {
            String base = "model.layers." + i + ".";
            String layerType = gemma4Config.layerTypes.get(i);
            int sharedSource = gemma4Config.getSharedKvSourceLayer(i);
            boolean kvSharedLayer = sharedSource >= 0;

            Gemma4CausalSelfAttention attention = new Gemma4CausalSelfAttention(
                    this,
                    i,
                    layerType,
                    kvSharedLayer,
                    gemma4Config.storesSharedKvState(i),
                    sharedSource,
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
                        quantize(weights.load(base + "post_per_layer_input_norm.weight"), qType), metricRegistry));
            }

            blocks[i] = new Gemma4TransformerBlock(
                    this,
                    i,
                    new RmsNorm(this, quantize(weights.load(base + "input_layernorm.weight"), qType), 1.0f, metricRegistry),
                    attention,
                    new RmsNorm(this, quantize(weights.load(base + "post_attention_layernorm.weight"), qType), 1.0f, metricRegistry),
                    new RmsNorm(this, quantize(weights.load(base + "pre_feedforward_layernorm.weight"), qType), 1.0f, metricRegistry),
                    mlp,
                    new RmsNorm(this, quantize(weights.load(base + "post_feedforward_layernorm.weight"), qType), 1.0f, metricRegistry),
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
            Optional<Consumer<java.util.List<AbstractTensor>>> tensorReducer) {
        return withSharedKeyValues(() -> {
            AbstractTensor embedding = embedInput.batchInputsToEmbeddings(tokenIds, startPos);
            AbstractTensor[] perLayerInputs = computePerLayerInputs(tokenIds, embedding);
            return forwardGemma4(embedding, perLayerInputs, startPos, kvbuf, tensorReducer);
        });
    }

    @Override
    public AbstractTensor batchForward(int[] tokenIds, int startPos, KvBufferCache.KvBuffer kvbuf) {
        return batchForward(tokenIds, startPos, kvbuf, Optional.empty());
    }

    @Override
    public AbstractTensor forward(int tokenId, int pos, KvBufferCache.KvBuffer kvbuf,
            Optional<Consumer<java.util.List<AbstractTensor>>> tensorReducer) {
        return withSharedKeyValues(() -> {
            AbstractTensor embedding = embedInput.inputTokenToEmbedding(tokenId, pos);
            AbstractTensor[] perLayerInputs = computePerLayerInputs(new int[]{tokenId}, embedding);
            return forwardGemma4(embedding, perLayerInputs, pos, kvbuf, tensorReducer);
        });
    }

    @Override
    public AbstractTensor forward(AbstractTensor embedding, int startPos, KvBufferCache.KvBuffer kvbuf,
            Optional<Consumer<java.util.List<AbstractTensor>>> tensorReducer) {
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

    public SharedKeyValues getSharedKeyValues(int layerIndex) {
        Map<Integer, SharedKeyValues> current = sharedKeyValues.get();
        if (current == null || !current.containsKey(layerIndex)) {
            throw new IllegalStateException("Missing shared kv state for layer " + layerIndex);
        }
        return current.get(layerIndex);
    }

    public void putSharedKeyValues(int layerIndex, AbstractTensor key, AbstractTensor value) {
        Map<Integer, SharedKeyValues> current = sharedKeyValues.get();
        if (current == null) {
            throw new IllegalStateException("Shared kv state not initialized");
        }
        SharedKeyValues previous = current.remove(layerIndex);
        if (previous != null) {
            previous.close();
        }
        AbstractTensor keyCopy = makeDenseTensor(TensorShape.of(key.shape().first(), key.shape().last()));
        AbstractTensor valueCopy = makeDenseTensor(TensorShape.of(value.shape().first(), value.shape().last()));
        keyCopy.copyFrom(key, 0, 0, (int) key.size());
        valueCopy.copyFrom(value, 0, 0, (int) value.size());
        current.put(layerIndex, new SharedKeyValues(keyCopy, valueCopy));
    }

    private AbstractTensor forwardGemma4(AbstractTensor embedding, AbstractTensor[] perLayerInputs, int startPos,
            KvBufferCache.KvBuffer kvbuf, Optional<Consumer<java.util.List<AbstractTensor>>> tensorReducer) {
        for (int i = config.dctx().layerStart; i < config.dctx().layerEnd; i++) {
            int relativeLayer = i - config.dctx().layerStart;
            AbstractTensor ref = embedding;
            AbstractTensor perLayerInput = perLayerInputs == null ? null : perLayerInputs[i];
            embedding = ((Gemma4TransformerBlock) transformerBlocks[relativeLayer]).forward(embedding, perLayerInput,
                    startPos, kvbuf, tensorReducer);
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

    private synchronized void ensurePerLayerWeightsLoaded() {
        if (gemma4Config.hiddenSizePerLayerInput == null || perLayerEmbeddingTable != null) {
            return;
        }
        DType qType = modelQType.orElse(this.modelDType);
        perLayerEmbeddingTable = quantize(weights.load("model.embed_tokens_per_layer.weight"), workingDType);
        perLayerModelProjectionWeights = quantize(weights.load("model.per_layer_model_projection.weight"), qType);
        perLayerProjectionNormWeights = quantize(weights.load("model.per_layer_projection_norm.weight"), qType);
    }

    private AbstractTensor[] computePerLayerInputs(int[] tokenIds, AbstractTensor embeddings) {
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
            for (int b = 0; b < batchSize; b++) {
                try (AbstractTensor row = perLayerEmbeddingTable.slice(true, tokenIds[b])) {
                    tokenIdentity.copyFrom(row, 0, tokenIdentity.getOffset(b, 0), packedLength);
                }
            }
            configurableTensorProvider.get().scale(perLayerEmbeddingScale, tokenIdentity, 0, packedLength);

            configurableTensorProvider.get().dotProductChunk(projected, embeddings, perLayerModelProjectionWeights, 0,
                    gemma4Config.embeddingLength, 0, packedLength);
            configurableTensorProvider.get().scale(perLayerModelProjectionScale, projected, 0, packedLength);
            Gemma4RmsNormSupport.applyInPlace(projected, gemma4Config.numberOfLayers,
                    gemma4Config.hiddenSizePerLayerInput, gemma4Config.layerNormEps, perLayerProjectionNormWeights);
            configurableTensorProvider.get().accumulate(projected, tokenIdentity, 0, packedLength);
            configurableTensorProvider.get().scale(perLayerInputScale, projected, 0, packedLength);

            AbstractTensor[] split = new AbstractTensor[gemma4Config.numberOfLayers];
            for (int layer = 0; layer < gemma4Config.numberOfLayers; layer++) {
                split[layer] = makeDenseTensor(batchSize, gemma4Config.hiddenSizePerLayerInput);
                int offset = layer * gemma4Config.hiddenSizePerLayerInput;
                for (int b = 0; b < batchSize; b++) {
                    split[layer].copyFrom(projected, projected.getOffset(b, offset), split[layer].getOffset(b, 0),
                            gemma4Config.hiddenSizePerLayerInput);
                }
            }
            return split;
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
        Map<Integer, SharedKeyValues> previous = sharedKeyValues.get();
        Map<Integer, SharedKeyValues> current = new HashMap<>();
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
}
