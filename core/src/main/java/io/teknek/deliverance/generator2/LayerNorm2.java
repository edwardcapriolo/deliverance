package io.teknek.deliverance.generator2;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.Histogram;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor2.TensorRef;
import net.jafama.FastMath;

/** TensorRef port of {@code LayerNorm}. */
public class LayerNorm2 {
    protected final AbstractModel model;
    private final TensorRef bias;
    protected final TensorRef weights;
    private final String biasName;
    private final String weightName;
    protected final MetricRegistry metricReigstry;
    public final Histogram totalTime;

    public LayerNorm2(AbstractModel model, TensorRef bias, TensorRef weights, MetricRegistry parent) {
        this(model, bias, weights, parent, "layernorm.bias", "layernorm.weight");
    }

    public LayerNorm2(AbstractModel model, TensorRef bias, TensorRef weights, MetricRegistry parent,
            String biasName, String weightName) {
        this.model = model;
        this.bias = bias;
        this.weights = weights;
        this.biasName = biasName;
        this.weightName = weightName;
        this.metricReigstry = parent;
        this.totalTime = metricReigstry.histogram("layer_norm");
    }

    public TensorRef forward(TensorRef input) {
        Preconditions.checkArgument(input.shape().dims() == 2);
        int size = input.shape().last();
        Preconditions.checkArgument(size == model.getConfig().embeddingLength);
        return forward(input, 0, model.getConfig().embeddingLength);
    }

    public TensorRef forward(TensorRef input, int offset, int length) {
        long start = System.currentTimeMillis();
        TensorRef output = model.makeDenseTensorRef(input.shape());
        int limit = offset + length;
        try {
            performLayerNorm(input, output, weights, bias, model.getConfig().layerNormEps, offset, length,
                    model.getConfig().embeddingLength);
            long end = System.currentTimeMillis();
            totalTime.update(end - start);
            return output;
        } catch (RuntimeException | Error e) {
            output.close();
            throw e;
        }
    }

    public static void performLayerNorm(TensorRef input, TensorRef output, TensorRef weights, TensorRef bias,
            float eps, int offset, int length, int embeddingLength) {
        int batchSize = input.shape().first();
        int limit = offset + length;
        for (int row = 0; row < batchSize; row++) {
            performLayerNormRow(input, output, weights, bias, eps, offset, limit, embeddingLength, row);
        }
    }

    static void performLayerNormRow(TensorRef input, TensorRef output, TensorRef weights, TensorRef bias,
            float eps, int offset, int limit, int embeddingLength, int row) {
        float sum = 0;
        float sumSq = 0;
        for (int i = offset; i < limit; i++) {
            float value = input.get(row, i);
            sum += value;
            sumSq += value * value;
        }
        float mean = sum / embeddingLength;
        float variance = sumSq / embeddingLength - mean * mean;
        float invStddev = 1.0f / (float) FastMath.sqrt(variance + eps);
        for (int i = offset; i < limit; i++) {
            float value = (input.get(row, i) - mean) * invStddev * weights.get(0, i) + bias.get(0, i);
            output.set(value, row, i);
        }
    }
}
