package io.teknek.deliverance.tensor;

import com.google.common.base.Preconditions;
import net.jafama.FastMath;

public class VectorTensorMathUtils {
    public static void softMax(AbstractTensor x, int offset, int length) {
        Preconditions.checkArgument(x.shape().first() == 1);
        long size = offset + length;

        // find max value (for numerical stability)
        float max_val = x.get(0, offset);
        for (int i = offset + 1; i < size; i++) {
            if (x.get(0, i) > max_val) {
                max_val = x.get(0, i);
            }
        }
        // exp and sum
        float sum = 0.0f;
        for (int i = offset; i < size; i++) {
            x.set((float) FastMath.exp(x.get(0, i) - max_val), 0, i);
            sum += x.get(0, i);
        }
        // normalize
        for (int i = 0; i < size; i++) {
            x.set(x.get(0, i) / sum, 0, i);
        }
    }

    public static void l2normalize(AbstractTensor x) {
        float sum = 0.0f;
        for (int i = 0; i < x.shape().last(); i++) {
            float v = x.get(0, i);
            sum += v * v;
        }
        double magnitude = FastMath.sqrt(sum);
        for (int i = 0; i < x.shape().last(); i++)
            x.set((float) (x.get(0, i) / magnitude), 0, i);
    }
}
