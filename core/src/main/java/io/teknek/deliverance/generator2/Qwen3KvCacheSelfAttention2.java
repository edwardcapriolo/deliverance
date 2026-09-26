package io.teknek.deliverance.generator2;

import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.concurrent.ForkJoinTask;

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
        ForkJoinTask<?> queryTask = model.getPool().getUnderlying().submit(() -> {
            try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(),
                    "kvcacheselfattention2.q_norm").time()) {
                compositeOps.groupedRmsNormInPlace(query, localNumberOfHeads, headDimension,
                        config.layerNormEps, queryNormWeights);
            }
        });
        ForkJoinTask<?> keyTask = model.getPool().getUnderlying().submit(() -> {
            try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(),
                    "kvcacheselfattention2.k_norm").time()) {
                compositeOps.groupedRmsNormInPlace(key, localNumberOfKeyValueHeads, headDimension,
                        config.layerNormEps, keyNormWeights);
            }
        });
        queryTask.join();
        keyTask.join();
    }

    @Override
    protected void applyRotaryEmbedding(TensorRef query, TensorRef key, int startPosition) {
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(),
                "kvcacheselfattention2.rope").time()) {
            float[][] frequencies = config.ropeFreqs.orElseThrow(
                    () -> new IllegalStateException("Qwen3 configuration does not provide RoPE frequencies"));
            compositeOps.rotaryEmbedding(query, localNumberOfHeads, headDimension, startPosition, frequencies);
            compositeOps.rotaryEmbedding(key, localNumberOfKeyValueHeads, headDimension, startPosition, frequencies);
        }
    }
}
