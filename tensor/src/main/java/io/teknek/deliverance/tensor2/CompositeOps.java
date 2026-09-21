package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricName;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import net.jafama.FastMath;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

public final class CompositeOps {
    private static final VectorSpecies<Float> F32_SPECIES = FloatVector.SPECIES_PREFERRED;
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

    /** Applies configured RoPE to the contiguous query/key heads in a TensorRef. */
    public void rotaryEmbedding(TensorRef target, int heads, int headSize, int startPosition,
            float[][] ropeFrequencies) {
        Preconditions.checkArgument(target != null, "RoPE target must be set");
        Preconditions.checkArgument(target.dims() == 2, "RoPE target must be 2D");
        Preconditions.checkArgument(heads >= 0 && headSize > 0 && (headSize & 1) == 0,
                "RoPE head shape is invalid");
        Preconditions.checkArgument(startPosition >= 0, "RoPE start position must be non-negative");
        int headPiece = headSize / 2;
        int batchSize = target.shape().first();
        Preconditions.checkArgument(target.shape().last() >= heads * headSize,
                "RoPE target does not contain all heads");
        Preconditions.checkArgument(ropeFrequencies != null
                        && ropeFrequencies.length >= (startPosition + batchSize) * headPiece,
                "RoPE frequencies do not cover the target positions");
        for (int row = 0; row < batchSize; row++) {
            int positionOffset = (startPosition + row) * headPiece;
            for (int head = 0; head < heads; head++) {
                int offset = head * headSize;
                for (int dimension = 0; dimension < headPiece; dimension++) {
                    float x0 = target.get(row, offset + dimension);
                    float x1 = target.get(row, offset + dimension + headPiece);
                    float[] frequency = ropeFrequencies[positionOffset + dimension];
                    float cosine = frequency[0];
                    float sine = frequency[1];
                    target.set(x0 * cosine - x1 * sine, row, offset + dimension);
                    target.set(x0 * sine + x1 * cosine, row, offset + dimension + headPiece);
                }
            }
        }
    }

    /** Applies F32 RMSNorm using vectorized row reductions and scaling. */
    public void rmsNorm(TensorRef output, TensorRef input, TensorRef weights, float epsilon,
            float weightAdjustment) {
        Preconditions.checkArgument(output.dType() == io.teknek.deliverance.DType.F32
                        && input.dType() == io.teknek.deliverance.DType.F32
                        && weights.dType() == io.teknek.deliverance.DType.F32,
                "F32 RMSNorm requires F32 tensors");
        Preconditions.checkArgument(output.shape().equals(input.shape()), "RMSNorm output shape must match input");
        Preconditions.checkArgument(weights.dims() == 2 && weights.shape().first() == 1
                        && weights.shape().last() >= input.shape().last(),
                "RMSNorm weights must be [1, inputWidth]");
        int rows = input.shape().first();
        int columns = input.shape().last();
        MemorySegment inputMemory = input.memorySegment();
        MemorySegment outputMemory = output.memorySegment();
        MemorySegment weightMemory = weights.memorySegment();
        int upper = F32_SPECIES.loopBound(columns);
        for (int row = 0; row < rows; row++) {
            long inputBase = input.memorySegmentOffset(input.shape().getOffset(row, 0));
            long outputBase = output.memorySegmentOffset(output.shape().getOffset(row, 0));
            long weightBase = weights.memorySegmentOffset(weights.shape().getOffset(0, 0));
            FloatVector sum = FloatVector.zero(F32_SPECIES);
            for (int column = 0; column < upper; column += F32_SPECIES.length()) {
                FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, inputMemory,
                        inputBase + (long) column * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                sum = values.fma(values, sum);
            }
            float sumSquares = sum.reduceLanes(VectorOperators.ADD);
            for (int column = upper; column < columns; column++) {
                float value = inputMemory.get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                        inputBase + (long) column * Float.BYTES);
                sumSquares += value * value;
            }
            float inverseRms = (float) (1.0 / StrictMath.sqrt(sumSquares / columns + epsilon));
            FloatVector inverse = FloatVector.broadcast(F32_SPECIES, inverseRms);
            FloatVector adjustment = FloatVector.broadcast(F32_SPECIES, weightAdjustment);
            for (int column = 0; column < upper; column += F32_SPECIES.length()) {
                long inputAddress = inputBase + (long) column * Float.BYTES;
                FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, inputMemory, inputAddress,
                        ByteOrder.LITTLE_ENDIAN);
                FloatVector scales = FloatVector.fromMemorySegment(F32_SPECIES, weightMemory,
                        weightBase + (long) column * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
                values.mul(inverse).mul(scales.add(adjustment)).intoMemorySegment(outputMemory,
                        outputBase + (long) column * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            }
            if (upper < columns) {
                VectorMask<Float> mask = F32_SPECIES.indexInRange(upper, columns);
                long inputAddress = inputBase + (long) upper * Float.BYTES;
                FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, inputMemory, inputAddress,
                        ByteOrder.LITTLE_ENDIAN, mask);
                FloatVector scales = FloatVector.fromMemorySegment(F32_SPECIES, weightMemory,
                        weightBase + (long) upper * Float.BYTES, ByteOrder.LITTLE_ENDIAN, mask);
                values.mul(inverse).mul(scales.add(adjustment)).intoMemorySegment(outputMemory,
                        outputBase + (long) upper * Float.BYTES, ByteOrder.LITTLE_ENDIAN, mask);
            }
        }
    }

    /** Applies vectorized SiLU to an F32 tensor in place. */
    public void silu(TensorRef target) {
        Preconditions.checkArgument(target.dType() == io.teknek.deliverance.DType.F32,
                "F32 SiLU requires an F32 tensor");
        MemorySegment memory = target.memorySegment();
        int rows = target.shape().first();
        int columns = target.shape().last();
        int upper = F32_SPECIES.loopBound(columns);
        for (int row = 0; row < rows; row++) {
            long base = target.memorySegmentOffset(target.shape().getOffset(row, 0));
            for (int column = 0; column < upper; column += F32_SPECIES.length()) {
                long address = base + (long) column * Float.BYTES;
                FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, memory, address,
                        ByteOrder.LITTLE_ENDIAN);
                FloatVector result = values.div(FloatVector.broadcast(F32_SPECIES, 1.0f)
                        .add(values.neg().lanewise(VectorOperators.EXP)));
                result.intoMemorySegment(memory, address, ByteOrder.LITTLE_ENDIAN);
            }
            if (upper < columns) {
                VectorMask<Float> mask = F32_SPECIES.indexInRange(upper, columns);
                long address = base + (long) upper * Float.BYTES;
                FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, memory, address,
                        ByteOrder.LITTLE_ENDIAN, mask);
                FloatVector result = values.div(FloatVector.broadcast(F32_SPECIES, 1.0f)
                        .add(values.neg().lanewise(VectorOperators.EXP)));
                result.intoMemorySegment(memory, address, ByteOrder.LITTLE_ENDIAN, mask);
            }
        }
    }

    /** Adds a residual tensor into an F32 target using vector operations. */
    public void addResidual(TensorRef target, TensorRef residual, float multiplier) {
        Preconditions.checkArgument(target.dType() == io.teknek.deliverance.DType.F32
                        && residual.dType() == io.teknek.deliverance.DType.F32,
                "F32 residuals require F32 tensors");
        Preconditions.checkArgument(target.shape().equals(residual.shape()), "Residual shapes must match");
        MemorySegment targetMemory = target.memorySegment();
        MemorySegment residualMemory = residual.memorySegment();
        int rows = target.shape().first();
        int columns = target.shape().last();
        int upper = F32_SPECIES.loopBound(columns);
        FloatVector factor = FloatVector.broadcast(F32_SPECIES, multiplier);
        for (int row = 0; row < rows; row++) {
            long targetBase = target.memorySegmentOffset(target.shape().getOffset(row, 0));
            long residualBase = residual.memorySegmentOffset(residual.shape().getOffset(row, 0));
            for (int column = 0; column < upper; column += F32_SPECIES.length()) {
                long targetAddress = targetBase + (long) column * Float.BYTES;
                long residualAddress = residualBase + (long) column * Float.BYTES;
                FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, targetMemory, targetAddress,
                        ByteOrder.LITTLE_ENDIAN);
                FloatVector additions = FloatVector.fromMemorySegment(F32_SPECIES, residualMemory, residualAddress,
                        ByteOrder.LITTLE_ENDIAN);
                values.add(additions.mul(factor)).intoMemorySegment(targetMemory, targetAddress,
                        ByteOrder.LITTLE_ENDIAN);
            }
            if (upper < columns) {
                VectorMask<Float> mask = F32_SPECIES.indexInRange(upper, columns);
                long targetAddress = targetBase + (long) upper * Float.BYTES;
                long residualAddress = residualBase + (long) upper * Float.BYTES;
                FloatVector values = FloatVector.fromMemorySegment(F32_SPECIES, targetMemory, targetAddress,
                        ByteOrder.LITTLE_ENDIAN, mask);
                FloatVector additions = FloatVector.fromMemorySegment(F32_SPECIES, residualMemory, residualAddress,
                        ByteOrder.LITTLE_ENDIAN, mask);
                values.add(additions.mul(factor)).intoMemorySegment(targetMemory, targetAddress,
                        ByteOrder.LITTLE_ENDIAN, mask);
            }
        }
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
