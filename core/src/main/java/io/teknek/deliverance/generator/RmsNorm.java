package io.teknek.deliverance.generator;

import com.codahale.metrics.Histogram;
import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import net.jafama.FastMath;

import java.time.Duration;

public class RmsNorm extends LayerNorm {
    private final float weightAdjustment;
    protected Timer totalTime;

    public RmsNorm(AbstractModel m, AbstractTensor weights, MetricRegistry metricRegistry) {
        this(m, weights, 0.0f, metricRegistry);
        totalTime = metricReigstry.timer("rms_norm");
    }

    public RmsNorm(AbstractModel m, AbstractTensor weights, float weightAdjustment, MetricRegistry metricRegistry) {
        super(m, null, weights, metricRegistry);
        totalTime = metricReigstry.timer("rms_norm");
        this.weightAdjustment = weightAdjustment;
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, int offset, int length) {
        long start = System.currentTimeMillis();
        int batchSize = input.shape().first();
        AbstractTensor output = model.makeDenseTensor(input.shape());
        int limit = offset + length;
        for (int b = 0; b < batchSize; b++) {
            double ss = 0.0f;
            for (int j = offset; j < limit; j++) {
                float v = input.get(b, j);
                ss += v * v;
            }
            //originally normalizaing over the enter length
            //ss /= model.getConfig().embeddingLength;
            ss /= length;
            ss += model.getConfig().layerNormEps;
            ss = (1.0 / FastMath.sqrt(ss));
            for (int j = offset; j < limit; j++) {
                output.set((weightAdjustment + weights.get(0, j)) * ((float) ss * input.get(b, j)), b, j);
            }
        }
        long end = System.currentTimeMillis();
        totalTime.update(Duration.ofMillis(end-start));
        return output;
    }
}
