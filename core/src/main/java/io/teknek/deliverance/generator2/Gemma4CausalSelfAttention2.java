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
    protected void normalizeQueryKey(TensorRef queryStates, TensorRef keyStates) {
        normalize(queryStates, queryNormWeights, numberOfHeads);
        normalize(keyStates, keyNormWeights, numberOfKeyValueHeads);
    }

    @Override
    protected void normalizeValue(TensorRef valueStates) {
        // HF's v_norm has no learned scale, but it still performs RMS normalization.
        normalize(valueStates, null, numberOfKeyValueHeads);
    }

    @Override
    protected TensorRef[] sharedKeyValues(TensorRef[] qkv, KvCacheSession kvSession, int startPosition) {
        io.teknek.deliverance.model.gemma4.Gemma4Model.V2SharedKeyValues sharedKvStates =
                kvSession.getOrAttach(io.teknek.deliverance.model.gemma4.Gemma4Model.V2SharedKeyValues.class,
                        io.teknek.deliverance.model.gemma4.Gemma4Model.V2SharedKeyValues::new);
        if (sharedKvSource < 0) {
            io.teknek.deliverance.model.gemma4.Gemma4Model gemma4 =
                    (io.teknek.deliverance.model.gemma4.Gemma4Model) model;
            sharedKvStates.restorePrefix(layerIndex, kvSession, startPosition, gemma4.getLighter());
            sharedKvStates.store(layerIndex, startPosition, qkv[1], qkv[2], gemma4.getLighter());
            return qkv;
        }
        TensorRef keyStates = sharedKvStates.get(sharedKvSource, true);
        TensorRef valueStates = sharedKvStates.get(sharedKvSource, false);
        if (qkv[1] != null && qkv[1] != keyStates) qkv[1].close();
        if (qkv[2] != null && qkv[2] != valueStates && qkv[2] != qkv[1]) qkv[2].close();
        qkv[1] = keyStates;
        qkv[2] = valueStates;
        return qkv;
    }

    @Override
    protected boolean writesCache(CacheExecutionMode mode) {
        return sharedKvSource < 0 && super.writesCache(mode);
    }

    @Override
    protected void closeAll(TensorRef[] tensors) {
        if (sharedKvSource >= 0) {
            if (tensors[0] != null) tensors[0].close();
            return;
        }
        super.closeAll(tensors);
    }

    @Override
    protected void applyRotaryEmbedding(TensorRef queryStates, TensorRef keyStates, int startPosition) {
        compositeOps.rotaryEmbedding(queryStates, numberOfHeads, headSize, startPosition, inverseRopeFrequencies);
        compositeOps.rotaryEmbedding(keyStates, numberOfKeyValueHeads, headSize, startPosition, inverseRopeFrequencies);
    }

    private void normalize(TensorRef tensor, TensorRef weights, int heads) {
        compositeOps.groupedRmsNormInPlace(tensor, heads, headSize, config.layerNormEps, weights);
    }
}
