package io.teknek.deliverance.generator2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor.kv.CacheExecutionMode;
import io.teknek.deliverance.tensor.kv.KvCacheSession;

/**
 * TensorRef attention primitive for one Gemma4 layer type.
 *
 * <p>This class intentionally handles only the local projection geometry and Q/K RMS normalization. Shared KV
 * ownership and layer-specific masking are model-level concerns and will be added when Gemma4's V2 block is wired.</p>
 */
public final class Gemma4CausalSelfAttention2 extends KvCacheSelfAttention2 {
    private final int numberOfHeads;
    private final int numberOfKeyValueHeads;
    private final float[] inverseRopeFrequencies;
    private final TensorRef queryNormWeights;
    private final TensorRef keyNormWeights;
    private final int sharedKvSource;

    public Gemma4CausalSelfAttention2(AbstractModel model, int layerIndex,
            TensorRef queryAttentionWeights, TensorRef queryNormWeights,
            TensorRef keyAttentionWeights, TensorRef valueAttentionWeights, TensorRef keyNormWeights,
            TensorRef outputProjectionWeights, Lighter lighter, MetricRegistry metricRegistry,
            int numberOfHeads, int numberOfKeyValueHeads, int headSize, int attentionLength, int kvLength,
            float[] inverseRopeFrequencies, String queryWeightName, String keyWeightName,
            String valueWeightName, String outputWeightName) {
        this(model, layerIndex, queryAttentionWeights, queryNormWeights, keyAttentionWeights, valueAttentionWeights,
                keyNormWeights, outputProjectionWeights, lighter, metricRegistry, numberOfHeads,
                numberOfKeyValueHeads, headSize, attentionLength, kvLength, inverseRopeFrequencies, queryWeightName,
                keyWeightName, valueWeightName, outputWeightName, false, false, 0, layerIndex);
    }

    public Gemma4CausalSelfAttention2(AbstractModel model, int layerIndex,
            TensorRef queryAttentionWeights, TensorRef queryNormWeights,
            TensorRef keyAttentionWeights, TensorRef valueAttentionWeights, TensorRef keyNormWeights,
            TensorRef outputProjectionWeights, Lighter lighter, MetricRegistry metricRegistry,
            int numberOfHeads, int numberOfKeyValueHeads, int headSize, int attentionLength, int kvLength,
            float[] inverseRopeFrequencies, String queryWeightName, String keyWeightName,
            String valueWeightName, String outputWeightName, boolean slidingAttention,
            boolean bidirectionalAttention, int attentionWindow, int cacheLayerIndex) {
        super(model, layerIndex, queryAttentionWeights, keyAttentionWeights, valueAttentionWeights,
                outputProjectionWeights, lighter, metricRegistry, queryWeightName, keyWeightName, valueWeightName,
                outputWeightName, attentionLength, kvLength, numberOfHeads, numberOfKeyValueHeads, headSize,
                slidingAttention ? attentionWindow : 0, bidirectionalAttention, cacheLayerIndex, 1.0f);
        this.numberOfHeads = numberOfHeads;
        this.numberOfKeyValueHeads = numberOfKeyValueHeads;
        this.queryNormWeights = queryNormWeights;
        this.keyNormWeights = keyNormWeights;
        this.inverseRopeFrequencies = inverseRopeFrequencies.clone();
        this.sharedKvSource = cacheLayerIndex == layerIndex ? -1 : cacheLayerIndex;
    }

    @Override
    protected void normalizeQueryKey(TensorRef query, TensorRef key) {
        normalize(query, queryNormWeights, numberOfHeads);
        normalize(key, keyNormWeights, numberOfKeyValueHeads);
    }

    @Override
    protected void normalizeValue(TensorRef value) {
        normalize(value, null, numberOfKeyValueHeads);
    }

    @Override
    protected TensorRef[] sharedKeyValues(TensorRef[] qkv, KvCacheSession kvSession) {
        io.teknek.deliverance.model.gemma4.Gemma4Model.V2SharedKeyValues state =
                kvSession.getOrAttach(io.teknek.deliverance.model.gemma4.Gemma4Model.V2SharedKeyValues.class,
                        io.teknek.deliverance.model.gemma4.Gemma4Model.V2SharedKeyValues::new);
        if (sharedKvSource < 0) {
            state.store(layerIndex, qkv[1], qkv[2], ((io.teknek.deliverance.model.gemma4.Gemma4Model) model).getLighter());
            return qkv;
        }
        TensorRef sharedKey = state.get(sharedKvSource, true);
        TensorRef sharedValue = state.get(sharedKvSource, false);
        qkv[1].close();
        qkv[2].close();
        qkv[1] = sharedKey;
        qkv[2] = sharedValue;
        return qkv;
    }

    @Override
    protected boolean writesCache(CacheExecutionMode mode) {
        return sharedKvSource < 0 && super.writesCache(mode);
    }

    @Override
    protected void applyRotaryEmbedding(TensorRef query, TensorRef key, int startPosition) {
        compositeOps.rotaryEmbedding(query, numberOfHeads, headSize, startPosition, inverseRopeFrequencies);
        compositeOps.rotaryEmbedding(key, numberOfKeyValueHeads, headSize, startPosition, inverseRopeFrequencies);
    }

    private void normalize(TensorRef tensor, TensorRef weights, int heads) {
        compositeOps.groupedRmsNormInPlace(tensor, heads, headSize, config.layerNormEps, weights);
    }
}
