package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricName;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import net.jafama.FastMath;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

public final class CompositeOps {
    public static final String TENSOR_TYPE = Lighter.TENSOR_TYPE;
    public static final String LENGTH = Lighter.LENGTH;

    private final Lighter lighter;
    private final MetricRegistry metricRegistry;

    public CompositeOps(Lighter lighter, MetricRegistry metricRegistry) {
        this.lighter = Objects.requireNonNull(lighter, "lighter");
        this.metricRegistry = metricRegistry;
    }

    public CompositeOps(Lighter lighter) {
        this(lighter, null);
    }

    public void multiplyInPlace(MultiplyInPlace operation) {
        multiplyInPlace(operation, Map.of());
    }

    public void multiplyInPlace(MultiplyInPlace operation, Map<String, String> tags) {
        validateMultiplyInPlace(operation);
        run("tensor2.composite.multiply_in_place", operation.getTarget(), operation.getLength(), tags, () -> {
            try {
                lighter.scale(new Scale(operation.getFactor())
                        .target(operation.getTarget())
                        .offsetAndLength(operation.getOffset(), operation.getLength()), tags);
            } catch (IllegalStateException e) {
                throw new UnsupportedOperationException("No tensor operations support multiplyInPlace", e);
            }
        });
    }

    public void softMax(ScaledSoftMax operation) {
        scaledSoftMax(operation);
    }

    public void scaledSoftMax(ScaledSoftMax operation) {
        scaledSoftMax(operation, Map.of());
    }

    public void scaledSoftMax(ScaledSoftMax operation, Map<String, String> tags) {
        validateScaledSoftMax(operation);
        run("tensor2.composite.scaled_softmax", operation.getTarget(), operation.getLength(), tags, () -> {
            TensorRef target = operation.getTarget();
            int offset = operation.getOffset();
            int length = operation.getLength();
            int limit = offset + length;
            float maxVal = transformForAttentionSoftmax(target.underlying().get(0, offset), operation.getScale(),
                    operation.getSoftcap());
            for (int i = offset + 1; i < limit; i++) {
                float value = transformForAttentionSoftmax(target.underlying().get(0, i), operation.getScale(),
                        operation.getSoftcap());
                if (value > maxVal) {
                    maxVal = value;
                }
            }
            float sum = 0.0f;
            for (int i = offset; i < limit; i++) {
                float value = transformForAttentionSoftmax(target.underlying().get(0, i), operation.getScale(),
                        operation.getSoftcap());
                target.underlying().set((float) FastMath.exp(value - maxVal), 0, i);
                sum += target.underlying().get(0, i);
            }
            multiplyInPlace(new MultiplyInPlace(1.0f / sum).target(target).offsetAndLength(offset, length), tags);
        });
    }

    private void validateMultiplyInPlace(MultiplyInPlace operation) {
        TensorRef target = operation.getTarget();
        Preconditions.checkArgument(target != null, "Target tensor must be set");
        Preconditions.checkArgument(operation.getOffset() >= 0 && operation.getLength() >= 0
                && operation.getOffset() + operation.getLength() <= target.shape().last(),
                "Multiply window out of bounds");
    }

    private void validateScaledSoftMax(ScaledSoftMax operation) {
        TensorRef target = operation.getTarget();
        Preconditions.checkArgument(target != null, "Target tensor must be set");
        Preconditions.checkArgument(target.shape().first() == 1, "scaledSoftMax requires one row");
        Preconditions.checkArgument(operation.getOffset() >= 0 && operation.getLength() > 0
                && operation.getOffset() + operation.getLength() <= target.shape().last(),
                "Softmax window out of bounds");
    }

    private void run(String metricName, TensorRef target, int length, Map<String, String> tags, Runnable action) {
        if (metricRegistry == null) {
            action.run();
            return;
        }
        Map<String, String> metricTags = new HashMap<>(tags);
        metricTags.put(TENSOR_TYPE, target.dType().name());
        metricTags.put(LENGTH, String.valueOf(length));
        Timer timer = metricRegistry.timer(new MetricName(metricName + ".time", metricTags));
        long startNanos = System.nanoTime();
        action.run();
        metricRegistry.meter(new MetricName(metricName, metricTags)).mark();
        timer.update(System.nanoTime() - startNanos, TimeUnit.NANOSECONDS);
    }

    private static float transformForAttentionSoftmax(float value, float scale, Float softcap) {
        float scaled = value * scale;
        if (softcap == null) {
            return scaled;
        }
        return (float) FastMath.tanh(scaled / softcap) * softcap;
    }
}
