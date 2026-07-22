package io.teknek.deliverance.model.granitemoehybrid;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.generator.CausalSelfAttention;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

public class GraniteMoeHybridAttention extends CausalSelfAttention {

    public GraniteMoeHybridAttention(AbstractModel model, int layerIndex, AbstractTensor queryAttnWeights,
            AbstractTensor keyAttnWeights, AbstractTensor valueAttnWeights, AbstractTensor outputProjectionWeights,
            ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry) {
        super(model, layerIndex, queryAttnWeights, keyAttnWeights, valueAttnWeights, outputProjectionWeights,
                configurableTensorProvider, metricRegistry);
    }

}
