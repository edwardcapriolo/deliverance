package io.teknek.deliverance.generator2;

import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor2.TensorRef;
import net.jafama.FastMath;

/** TensorRef-native RMSNorm layer. */
public class RmsNorm2 extends LayerNorm2 {
    private final float weightAdjustment;
    private final float epsilon;
    protected Timer totalTime;

    public RmsNorm2(AbstractModel model, TensorRef weights, float weightAdjustment) {
        super(java.util.Objects.requireNonNull(model, "model"), null,
                java.util.Objects.requireNonNull(weights, "weights"), model.getMetricRegistry());
        this.weightAdjustment = weightAdjustment;
        this.epsilon = model.getConfig().layerNormEps;
        this.totalTime = metricReigstry.timer("rms_norm");
    }

    @Override
    public TensorRef forward(TensorRef input, int offset, int length) {
        long start = System.currentTimeMillis();
        TensorRef output = model.makeDenseTensorRef(input.shape());
        int limit = offset + length;
        try {
            applyRmsNorm(input, output, offset, length, limit);
            long end = System.currentTimeMillis();
            totalTime.update(java.time.Duration.ofMillis(end - start));
            return output;
        } catch (RuntimeException | Error e) {
            output.close();
            throw e;
        }
    }

    private void applyRmsNorm(TensorRef input, TensorRef output, int offset, int length, int limit) {
        for (int row = 0; row < input.shape().first(); row++) {
            double sumSquares = 0.0;
            for (int column = offset; column < limit; column++) {
                float value = input.get(row, column);
                sumSquares += value * value;
            }
            double scale = 1.0 / FastMath.sqrt((sumSquares / length) + epsilon);
            for (int column = offset; column < limit; column++) {
                output.set((weightAdjustment + weights.get(0, column)) * ((float) scale * input.get(row, column)),
                        row, column);
            }
        }
    }
}
