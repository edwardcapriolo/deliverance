package io.teknek.deliverance.generator;

import io.teknek.deliverance.tensor.AbstractTensor;
import net.jafama.FastMath;

public final class Gemma4RmsNormSupport {
    private Gemma4RmsNormSupport() {
    }

    public static void applyInPlace(AbstractTensor tensor, int groups, int groupSize, float eps, AbstractTensor weights) {
        int batchSize = tensor.shape().first();
        for (int b = 0; b < batchSize; b++) {
            for (int g = 0; g < groups; g++) {
                int offset = g * groupSize;
                double sumSquares = 0.0;
                for (int i = 0; i < groupSize; i++) {
                    float value = tensor.get(b, offset + i);
                    sumSquares += value * value;
                }
                double invRms = 1.0 / FastMath.sqrt((sumSquares / groupSize) + eps);
                for (int i = 0; i < groupSize; i++) {
                    float scaled = (float) (tensor.get(b, offset + i) * invRms);
                    if (weights != null) {
                        scaled *= weights.shape().dims() == 1 ? weights.get(i) : weights.get(0, i);
                    }
                    tensor.set(scaled, b, offset + i);
                }
            }
        }
    }
}
