package io.teknek.deliverance.generator2;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.dropwizard.metrics5.MetricRegistry;

/** Generic TensorRef KV-cache attention without Qwen-specific query/key RMSNorm. */
public final class CausalSelfAttention2 extends KvCacheSelfAttention2 {
    private final int headSize;
    private final int numberOfHeads;
    private final int numberOfKeyValueHeads;

    public CausalSelfAttention2(AbstractModel model, int layerIndex, TensorRef queryAttentionWeights,
            TensorRef keyAttentionWeights, TensorRef valueAttentionWeights, TensorRef outputProjectionWeights,
            Lighter lighter, MetricRegistry metricRegistry, String queryWeightName, String keyWeightName,
            String valueWeightName, String outputWeightName) {
        super(model, layerIndex, queryAttentionWeights, keyAttentionWeights, valueAttentionWeights,
                outputProjectionWeights, lighter, metricRegistry, queryWeightName, keyWeightName, valueWeightName,
                outputWeightName);
        this.headSize = config.headSize;
        this.numberOfHeads = model.getLocalNumberOfHeads();
        this.numberOfKeyValueHeads = model.getLocalNumberOfKeyValueHeads();
    }

    @Override
    protected void applyRotaryEmbedding(TensorRef query, TensorRef key, int startPosition) {
        float[][] frequencies = config.ropeFreqs.orElseThrow(
                () -> new IllegalStateException("Configuration does not provide RoPE frequencies"));
        compositeOps.rotaryEmbedding(query, numberOfHeads, headSize, startPosition, frequencies);
        compositeOps.rotaryEmbedding(key, numberOfKeyValueHeads, headSize, startPosition, frequencies);
    }
}
