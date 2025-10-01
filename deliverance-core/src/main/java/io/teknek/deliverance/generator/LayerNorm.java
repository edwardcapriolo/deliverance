package io.teknek.deliverance.generator;



import com.google.common.base.Preconditions;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import net.jafama.FastMath;

public class LayerNorm {

    protected final AbstractModel model;
    private final AbstractTensor bias;
    protected final AbstractTensor weights;

    public LayerNorm(AbstractModel m, AbstractTensor bias, AbstractTensor weights) {
        this.model = m;
        this.bias = bias;
        this.weights = weights;
    }

    public AbstractTensor forward(AbstractTensor input) {
        Preconditions.checkArgument(input.shape().dims() == 2);
        int size = input.shape().last();
        Preconditions.checkArgument(size == model.getConfig().embeddingLength);
        return forward(input, 0, model.getConfig().embeddingLength);
    }

    public AbstractTensor forward(AbstractTensor input, int offset, int length) {

        int batchSize = input.shape().first();
        AbstractTensor output = input.copyShape();

        for (int b = 0; b < batchSize; b++) {
            float sum = 0;
            float sumSq = 0;
            int limit = offset + length;
            for (int i = offset; i < limit; i++) {
                float v = input.get(b, i);
                sum += v;
                sumSq += v * v;
            }
            float mean = sum / model.getConfig().embeddingLength;
            float variance = sumSq / model.getConfig().embeddingLength - mean * mean;
            float invStddev = 1.0f / (float) FastMath.sqrt(variance + model.getConfig().layerNormEps);
            for (int i = offset; i < limit; i++) {
                float v = (input.get(b, i) - mean) * invStddev * weights.get(0, i) + bias.get(0, i);
                output.set(v, b, i);
            }
        }

        return output;
    }
}
