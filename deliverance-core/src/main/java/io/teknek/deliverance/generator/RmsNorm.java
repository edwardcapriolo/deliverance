package io.teknek.deliverance.generator;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import net.jafama.FastMath;

public class RmsNorm extends LayerNorm {
    private final float weightAdjustment;

    public RmsNorm(AbstractModel m, AbstractTensor weights) {
        this(m, weights, 0.0f);
    }

    public RmsNorm(AbstractModel m, AbstractTensor weights, float weightAdjustment) {
        super(m, null, weights);
        this.weightAdjustment = weightAdjustment;
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, int offset, int length) {
        int batchSize = input.shape().first();
        AbstractTensor output = model.makeDenseTensor(input.shape());
        int limit = offset + length;
        for (int b = 0; b < batchSize; b++) {
            double ss = 0.0f;
            for (int j = offset; j < limit; j++) {
                float v = input.get(b, j);
                ss += v * v;
            }
            ss /= this.model.getConfig().embeddingLength;
            ss += this.model.getConfig().layerNormEps;
            ss = (1.0 / FastMath.sqrt(ss));
            for (int j = offset; j < limit; j++) {
                output.set((weightAdjustment + weights.get(0, j)) * ((float) ss * input.get(b, j)), b, j);
            }
        }
        return output;
    }
}
