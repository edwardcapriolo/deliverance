package io.teknek.deliverance.generator;

import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

import java.util.concurrent.ForkJoinTask;

/** KVCache2 attention for Qwen3, including Q/K head RMSNorm before RoPE and cache write. */
public final class Qwen3KvCacheSelfAttention extends KvCacheSelfAttention {
    private final AbstractModel model;
    private final AbstractTensor qNormWeights;
    private final AbstractTensor kNormWeights;
    private final int headDim;
    private final int numberOfHeads;
    private final int numberOfKeyValueHeads;

    public Qwen3KvCacheSelfAttention(AbstractModel model, int layerIndex, AbstractTensor queryAttnWeights,
            AbstractTensor keyAttnWeights, AbstractTensor valueAttnWeights, AbstractTensor outputProjectionWeights,
            AbstractTensor qNormWeights, AbstractTensor kNormWeights,
            ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry,
            String queryWeightName, String keyWeightName, String valueWeightName, String outputWeightName) {
        super(model, layerIndex, queryAttnWeights, keyAttnWeights, valueAttnWeights, outputProjectionWeights,
                configurableTensorProvider, metricRegistry, queryWeightName, keyWeightName, valueWeightName,
                outputWeightName);
        this.model = model;
        this.qNormWeights = qNormWeights;
        this.kNormWeights = kNormWeights;
        this.headDim = model.getConfig().headSize;
        this.numberOfHeads = model.getLocalNumberOfHeads();
        this.numberOfKeyValueHeads = model.getLocalNumberOfKeyValueHeads();
    }

    @Override
    protected void normalizeQueryKey(AbstractTensor query, AbstractTensor key) {
        ForkJoinTask<?> queryTask = model.getPool().getUnderlying().submit(() -> {
            try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(),
                    "kvcacheselfattention.q_norm").time()) {
                Gemma4RmsNormSupport.applyInPlaceSimd(query, numberOfHeads, headDim, model.getConfig().layerNormEps,
                        qNormWeights);
            }
        });
        ForkJoinTask<?> keyTask = model.getPool().getUnderlying().submit(() -> {
            try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(),
                    "kvcacheselfattention.k_norm").time()) {
                Gemma4RmsNormSupport.applyInPlaceSimd(key, numberOfKeyValueHeads, headDim,
                        model.getConfig().layerNormEps, kNormWeights);
            }
        });
        queryTask.join();
        keyTask.join();
    }
}
