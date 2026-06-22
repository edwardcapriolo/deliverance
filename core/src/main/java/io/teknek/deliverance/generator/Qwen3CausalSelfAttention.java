package io.teknek.deliverance.generator;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

import java.util.Optional;

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
        super(model, layerIndex, queryAttnWeights, keyAttnWeights, valueAttnWeights, outputProjectionWeights,
                configurableTensorProvider, metricRegistry);
        this.model = model;
        this.qNormWeights = qNormWeights;
        this.kNormWeights = kNormWeights;
        this.headDim = model.getConfig().headSize;
        this.numberOfHeads = model.getLocalNumberOfHeads();
        this.numberOfKeyValueHeads = model.getLocalNumberOfKeyValueHeads();
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, int startPosition, KvBufferCache.KvBuffer kvMem,
            Optional<java.util.function.Consumer<java.util.List<AbstractTensor>>> tensorReducer) {
        // Qwen3 q/k head RMSNorm is implemented by the base class hook below.
        return super.forward(input, startPosition, kvMem, tensorReducer);
    }

    @Override
    protected void normalizeQueryKey(AbstractTensor query, AbstractTensor key) {
        Gemma4RmsNormSupport.applyInPlace(query, numberOfHeads, headDim, model.getConfig().layerNormEps, qNormWeights);
        Gemma4RmsNormSupport.applyInPlace(key, numberOfKeyValueHeads, headDim, model.getConfig().layerNormEps, kNormWeights);
    }
}
