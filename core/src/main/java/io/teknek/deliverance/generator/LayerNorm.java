package io.teknek.deliverance.generator;

import com.codahale.metrics.Histogram;
import com.google.common.base.Preconditions;
import io.teknek.deliverance.CausualWhisperer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import net.jafama.FastMath;
import com.codahale.metrics.MetricRegistry;
/*
* https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
* */
public class LayerNorm {

    protected final AbstractModel model;
    private final AbstractTensor bias;
    protected final AbstractTensor weights;
    protected final MetricRegistry metricReigstry;
    public final Histogram totalTime;

    public LayerNorm(AbstractModel m, AbstractTensor bias, AbstractTensor weights, MetricRegistry parent) {
        this.model = m;
        this.bias = bias;
        this.weights = weights;
        this.metricReigstry = parent;
        totalTime = metricReigstry.histogram("layer_norm");
    }

    public AbstractTensor forward(AbstractTensor input) {
        Preconditions.checkArgument(input.shape().dims() == 2);
        int size = input.shape().last();
        Preconditions.checkArgument(size == model.getConfig().embeddingLength);
        return forward(input, 0, model.getConfig().embeddingLength);
    }

    public AbstractTensor forward(AbstractTensor input, int offset, int length) {
        long start = System.currentTimeMillis();
        int batchSize = input.shape().first();
        AbstractTensor output = model.getTensorCache().getDirty(input.dType(), input.shape());
        for (int b = 0; b < batchSize; b++) {
            float sum = 0;
            float sumSq = 0;
            int limit = offset + length;
            if (b == 3) {
                CausualWhisperer.LOGGER.info("LayerNorm.forward batch {} loop offset {} to limit {}", b, offset, limit);
            }
            for (int i = offset; i < limit; i++) {
                float v = input.get(b, i);
                sum += v;
                sumSq += v * v;
            }
            float mean = sum / model.getConfig().embeddingLength;
            float variance = sumSq / model.getConfig().embeddingLength - mean * mean;
            float invStddev = 1.0f / (float) FastMath.sqrt(variance + model.getConfig().layerNormEps);
            if (b == 3) {
                CausualWhisperer.LOGGER.info("LayerNorm.forward sum {} sumSq {} mean {} variance {} invStdDev {} ",
                        sum, sumSq, mean, variance, invStddev);
            }
            for (int i = offset; i < limit; i++) {
                float v = (input.get(b, i) - mean) * invStddev * weights.get(0, i) + bias.get(0, i);
                output.set(v, b, i);
            }
        }
        long end = System.currentTimeMillis();
        totalTime.update(end-start);
        return output;
    }
}
