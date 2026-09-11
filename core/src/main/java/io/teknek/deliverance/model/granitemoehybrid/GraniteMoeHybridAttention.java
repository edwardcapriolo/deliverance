package io.teknek.deliverance.model.granitemoehybrid;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.generator.KvCacheSelfAttention;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

public class GraniteMoeHybridAttention extends KvCacheSelfAttention {

    public GraniteMoeHybridAttention(AbstractModel model, int layerIndex, AbstractTensor queryAttnWeights,
            AbstractTensor keyAttnWeights, AbstractTensor valueAttnWeights, AbstractTensor outputProjectionWeights,
            ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry, String queryWeightName,
            String keyWeightName, String valueWeightName, String outputWeightName) {
        super(model, layerIndex, queryAttnWeights, keyAttnWeights, valueAttnWeights, outputProjectionWeights,
                configurableTensorProvider, metricRegistry, queryWeightName, keyWeightName, valueWeightName,
                outputWeightName);
    }

}
