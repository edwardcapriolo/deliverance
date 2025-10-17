package io.teknek.deliverance.math;


import com.google.common.base.Preconditions;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.operations.BiIntConsumer;
import net.jafama.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VectorMath {

    private static final Logger logger = LoggerFactory.getLogger(VectorMath.class);

    public static void pfor(int start, int end, IntConsumer action) {
        PhysicalCoreTuningExecutor.instance.get().execute(() -> IntStream.range(start, end).parallel().forEach(action));
    }

    public static void pchunk(int offset, int length, BiIntConsumer action, int splitSize) {
        int splits = Math.min(length, splitSize);
        int chunkSize = length / splits;

        if (splits == 1) {
            action.accept(offset, length);
        } else {
            int remainder = length % chunkSize;

            int fsplits = splits;
            int fchunkSize = chunkSize;
            int fremainder = remainder;

            PhysicalCoreTuningExecutor.instance.get()
                    .execute(
                            () -> IntStream.range(0, fsplits)
                                    .parallel()
                                    .forEach(
                                            i -> action.accept(
                                                    offset + (i * fchunkSize),
                                                    fremainder > 0 && i == fsplits - 1 ? fchunkSize + fremainder : fchunkSize
                                            )
                                    )
                    );
        }
    }

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
