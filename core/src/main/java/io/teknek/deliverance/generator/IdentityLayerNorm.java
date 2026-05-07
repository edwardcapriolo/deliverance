package io.teknek.deliverance.generator;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;

public class IdentityLayerNorm extends LayerNorm {
    public IdentityLayerNorm(AbstractModel model, MetricRegistry metricRegistry) {
        super(model, null, null, metricRegistry);
    }

    @Override
    public AbstractTensor forward(AbstractTensor input) {
        AbstractTensor output = model.makeDenseTensor(input.shape());
        output.copyFrom(input, 0, 0, (int) input.size());
        return output;
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, int offset, int length) {
        return forward(input);
    }
}
