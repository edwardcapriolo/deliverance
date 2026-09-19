package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricName;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import net.jafama.FastMath;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CompositeOpsScaledSoftMaxTest {

    @Test
    void softMaxRespectsOffsetWindow() {
        MetricRegistry metrics = new MetricRegistry();
        Lighter lighter = new Lighter(metrics);
        CompositeOps ops = new CompositeOps(lighter, metrics);
        try (TensorRef target = lighter.allocate(DType.F32, TensorShape.of(1, 5))) {
            target.underlying().set(99.0f, 0, 0);
            target.underlying().set(88.0f, 0, 1);
            target.underlying().set(1.0f, 0, 2);
            target.underlying().set(2.0f, 0, 3);
            target.underlying().set(3.0f, 0, 4);

            ops.scaledSoftMax(new ScaledSoftMax(1.0f).target(target).offsetAndLength(2, 3));

            assertEquals(99.0f, target.underlying().get(0, 0), 0.000001f);
            assertEquals(88.0f, target.underlying().get(0, 1), 0.000001f);
            float sum = target.underlying().get(0, 2) + target.underlying().get(0, 3)
                    + target.underlying().get(0, 4);
            assertEquals(1.0f, sum, 0.000001f);
            assertTrue(target.underlying().get(0, 4) > target.underlying().get(0, 3));
            assertTrue(target.underlying().get(0, 3) > target.underlying().get(0, 2));
        }
        assertEquals(1, metrics.meter(new MetricName("tensor2.composite.scaled_softmax",
                java.util.Map.of(CompositeOps.TENSOR_TYPE, DType.F32.name(), CompositeOps.LENGTH, "3"))).getCount());
    }

    @Test
    void scaledSoftMaxMatchesReferenceWithSoftcap() {
        Lighter lighter = new Lighter();
        CompositeOps ops = new CompositeOps(lighter);
        try (TensorRef target = lighter.allocate(DType.F32, TensorShape.of(1, 7))) {
            float[] expected = new float[7];
            for (int i = 0; i < 7; i++) {
                float value = (i - 3) * 1.75f;
                target.underlying().set(value, 0, i);
                expected[i] = value;
            }

            reference(expected, 1, 5, 0.25f, 1.5f);
            ops.scaledSoftMax(new ScaledSoftMax(0.25f).target(target).offsetAndLength(1, 5).softcap(1.5f));

            for (int i = 0; i < 7; i++) {
                assertEquals(expected[i], target.underlying().get(0, i), 1.0e-6f, "column=" + i);
            }
        }
    }

    @Test
    void scaledSoftMaxMatchesReferenceWithoutSoftcap() {
        Lighter lighter = new Lighter();
        CompositeOps ops = new CompositeOps(lighter);
        try (TensorRef target = lighter.allocate(DType.F32, TensorShape.of(1, 5))) {
            float[] expected = new float[5];
            for (int i = 0; i < 5; i++) {
                float value = (i - 2) * 1.25f;
                target.underlying().set(value, 0, i);
                expected[i] = value;
            }

            reference(expected, 0, 5, 0.5f, null);
            ops.scaledSoftMax(new ScaledSoftMax(0.5f).target(target).offsetAndLength(0, 5));

            for (int i = 0; i < 5; i++) {
                assertEquals(expected[i], target.underlying().get(0, i), 1.0e-6f, "column=" + i);
            }
        }
    }

    private static void reference(float[] values, int offset, int length, float scale, Float softcap) {
        int limit = offset + length;
        float max = transform(values[offset], scale, softcap);
        for (int i = offset + 1; i < limit; i++) {
            max = Math.max(max, transform(values[i], scale, softcap));
        }
        float sum = 0.0f;
        for (int i = offset; i < limit; i++) {
            values[i] = (float) FastMath.exp(transform(values[i], scale, softcap) - max);
            sum += values[i];
        }
        for (int i = offset; i < limit; i++) {
            values[i] /= sum;
        }
    }

    private static float transform(float value, float scale, Float softcap) {
        float scaled = value * scale;
        if (softcap == null) {
            return scaled;
        }
        return (float) FastMath.tanh(scaled / softcap) * softcap;
    }
}
