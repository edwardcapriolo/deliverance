package io.teknek.deliverance.generator2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.generator.Gemma4RmsNormSupport;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor2.TensorRefBackedTensor;

/** Qwen3 KV-cache attention with query/key RMSNorm hooks. */
public final class Qwen3KvCacheSelfAttention2 extends KvCacheSelfAttention2 {
    private final TensorRef queryNormWeights;
    private final TensorRef keyNormWeights;
    private final int headDimension;
    private final int localNumberOfHeads;
    private final int localNumberOfKeyValueHeads;

    public Qwen3KvCacheSelfAttention2(AbstractModel model, int layerIndex, TensorRef queryAttentionWeights,
            TensorRef keyAttentionWeights, TensorRef valueAttentionWeights, TensorRef outputProjectionWeights,
            TensorRef queryNormWeights, TensorRef keyNormWeights, Lighter lighter, MetricRegistry metricRegistry,
            String queryWeightName, String keyWeightName, String valueWeightName, String outputWeightName) {
        super(model, layerIndex, queryAttentionWeights, keyAttentionWeights, valueAttentionWeights,
                outputProjectionWeights, lighter, metricRegistry, queryWeightName, keyWeightName, valueWeightName,
                outputWeightName);
        this.queryNormWeights = java.util.Objects.requireNonNull(queryNormWeights, "queryNormWeights");
        this.keyNormWeights = java.util.Objects.requireNonNull(keyNormWeights, "keyNormWeights");
        this.headDimension = config.headSize;
        this.localNumberOfHeads = model.getLocalNumberOfHeads();
        this.localNumberOfKeyValueHeads = model.getLocalNumberOfKeyValueHeads();
    }

    @Override
    protected void normalizeQueryKey(TensorRef query, TensorRef key) {
        // This is a temporary boundary to the existing SIMD RMSNorm kernel. The adapters are borrowed
        // views; model-owned TensorRefs remain owned by the model and must not be closed here.
        AbstractTensor queryTensor = new TensorRefBackedTensor(query);
        AbstractTensor keyTensor = new TensorRefBackedTensor(key);
        AbstractTensor queryWeights = new TensorRefBackedTensor(queryNormWeights);
        AbstractTensor keyWeights = new TensorRefBackedTensor(keyNormWeights);
        Gemma4RmsNormSupport.applyInPlaceSimd(queryTensor, localNumberOfHeads, headDimension,
                config.layerNormEps, queryWeights);
        Gemma4RmsNormSupport.applyInPlaceSimd(keyTensor, localNumberOfKeyValueHeads, headDimension,
                config.layerNormEps, keyWeights);
    }

    @Override
    protected void applyRotaryEmbedding(TensorRef query, TensorRef key, int startPosition) {
        float[][] frequencies = config.ropeFreqs.orElseThrow(
                () -> new IllegalStateException("Qwen3 configuration does not provide RoPE frequencies"));
        compositeOps.rotaryEmbedding(query, localNumberOfHeads, headDimension, startPosition, frequencies);
        compositeOps.rotaryEmbedding(key, localNumberOfKeyValueHeads, headDimension, startPosition, frequencies);
    }
}
