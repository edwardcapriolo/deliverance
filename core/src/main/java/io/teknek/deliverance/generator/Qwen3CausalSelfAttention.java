package io.teknek.deliverance.generator;

import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensorlib.TensorPlan;

import java.util.Optional;
import java.util.concurrent.ForkJoinTask;

public class Qwen3CausalSelfAttention extends CausalSelfAttention {
    private final AbstractModel model;
    private final AbstractTensor qNormWeights;
    private final AbstractTensor kNormWeights;
    private final int headDim;
    private final int numberOfHeads;
    private final int numberOfKeyValueHeads;

    public Qwen3CausalSelfAttention(
            AbstractModel model,
            int layerIndex,
            AbstractTensor queryAttnWeights,
            AbstractTensor keyAttnWeights,
            AbstractTensor valueAttnWeights,
            AbstractTensor outputProjectionWeights,
            AbstractTensor qNormWeights,
            AbstractTensor kNormWeights,
            ConfigurableTensorProvider configurableTensorProvider,
            MetricRegistry metricRegistry
    ) {
        this(model, layerIndex, queryAttnWeights, keyAttnWeights, valueAttnWeights, outputProjectionWeights,
                qNormWeights, kNormWeights, configurableTensorProvider, metricRegistry, null, null, null, null);
    }

    /** Variant carrying base tensor names for LoRA runtime hot-swap -- see step 4 plan Section 4.1. */
    public Qwen3CausalSelfAttention(
            AbstractModel model,
            int layerIndex,
            AbstractTensor queryAttnWeights,
            AbstractTensor keyAttnWeights,
            AbstractTensor valueAttnWeights,
            AbstractTensor outputProjectionWeights,
            AbstractTensor qNormWeights,
            AbstractTensor kNormWeights,
            ConfigurableTensorProvider configurableTensorProvider,
            MetricRegistry metricRegistry,
            String queryWeightName,
            String keyWeightName,
            String valueWeightName,
            String outputWeightName
    ) {
        super(model, layerIndex, queryAttnWeights, keyAttnWeights, valueAttnWeights, outputProjectionWeights,
                configurableTensorProvider, metricRegistry, queryWeightName, keyWeightName, valueWeightName, outputWeightName);
        this.model = model;
        this.qNormWeights = qNormWeights;
        this.kNormWeights = kNormWeights;
        this.headDim = model.getConfig().headSize;
        this.numberOfHeads = model.getLocalNumberOfHeads();
        this.numberOfKeyValueHeads = model.getLocalNumberOfKeyValueHeads();
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, int startPosition, KvBufferCache.KvBuffer kvMem,
            Optional<java.util.function.Consumer<java.util.List<AbstractTensor>>> tensorReducer, ForwardPhase phase) {
        // Qwen3 q/k head RMSNorm is implemented by the base class hook below.
        return super.forward(input, startPosition, kvMem, tensorReducer, phase);
    }

    @Override
    protected void normalizeQueryKey(AbstractTensor query, AbstractTensor key) {

        //TensorPlan tp = new TensorPlan(model.primaryTensorOperations(), model.getPool(), model.getMetricRegistry());
        ForkJoinTask<?> queryTask = model.getPool().getUnderlying().submit( () -> {
            try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(),
                "causalselfattention.q_norm").time()) {
                Gemma4RmsNormSupport.applyInPlaceSimd(query, numberOfHeads, headDim, model.getConfig().layerNormEps,
                    qNormWeights);
        } } );
        ForkJoinTask<?> keyTask = model.getPool().getUnderlying().submit( () -> {
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(),
                "causalselfattention.k_norm").time()) {
            Gemma4RmsNormSupport.applyInPlaceSimd(key, numberOfKeyValueHeads, headDim, model.getConfig().layerNormEps,
                    kNormWeights);
        } } );
        queryTask.join();
        keyTask.join();
    }
}
