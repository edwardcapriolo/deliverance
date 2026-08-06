package io.teknek.deliverance.generator;

import io.teknek.deliverance.math.FloatConversions;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import com.google.common.base.Preconditions;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import net.jafama.FastMath;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

public final class Gemma4RmsNormSupport {
    private static final VectorSpecies<Float> FLOAT_SPECIES = FloatVector.SPECIES_PREFERRED;

    private Gemma4RmsNormSupport() {
    }

    /**
     * Applies independent RMSNorm groups in place across each row of {@code tensor}.
     *
     * <p>The tensor is interpreted as {@code [batchSize, groups * groupSize]}. For every batch row and every group, this
     * normalizes the contiguous slice {@code tensor[row, group * groupSize .. (group + 1) * groupSize)}:</p>
     *
     * <pre>{@code
     * invRms = 1 / sqrt(mean(slice[i]^2) + eps)
     * slice[i] = slice[i] * invRms * weight[i]
     * }</pre>
     *
     * <p>For Qwen3 query/key normalization, {@code groups} is the number of attention heads (or key/value heads) and
     * {@code groupSize} is the per-head dimension. That means each attention head is normalized independently rather than
     * normalizing the whole projected query/key row as one vector.</p>
     *
     * <p>{@code weights} is optional. When present, it may be either a 1-D vector of length {@code groupSize} or a
     * single-row 2-D tensor {@code [1, groupSize]}. The same weights are reused for every group.</p>
     */
    public static void applyInPlace(AbstractTensor tensor, int groups, int groupSize, float eps, AbstractTensor weights) {
        validateWeights(weights, groupSize);
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
                        scaled *= weights.get(0, i);
                    }
                    tensor.set(scaled, b, offset + i);
                }
            }
        }
    }

    public static void applyInPlaceSimd(AbstractTensor tensor, int groups, int groupSize, float eps,
            AbstractTensor weights) {
        validateWeights(weights, groupSize);
        if (!(tensor instanceof FloatBufferTensor floatTensor) || !supportsSimdWeights(weights)) {
            applyInPlace(tensor, groups, groupSize, eps, weights);
            return;
        }
        float[] decodedWeights = weights instanceof BFloat16BufferTensor ? decodeBf16Weights(weights, groupSize) : null;
        int batchSize = (int) tensor.shape().first();
        for (int row = 0; row < batchSize; row++) {
            for (int group = 0; group < groups; group++) {
                int offset = group * groupSize;
                int base = floatTensor.getOffset(row, offset);
                double sumSquares = sumSquares(floatTensor, base, groupSize);
                float invRms = (float) (1.0 / FastMath.sqrt((sumSquares / groupSize) + eps));
                scaleGroup(floatTensor, weights, decodedWeights, base, groupSize, invRms);
            }
        }
    }

    private static boolean supportsSimdWeights(AbstractTensor weights) {
        return weights == null || weights instanceof FloatBufferTensor || weights instanceof BFloat16BufferTensor;
    }

    private static void validateWeights(AbstractTensor weights, int groupSize) {
        if (weights == null) {
            return;
        }
        Preconditions.checkArgument(weights.dims() == 2, "RMSNorm weights must be [1, groupSize]");
        Preconditions.checkArgument(weights.shape().first() == 1, "RMSNorm weights must have one row");
        Preconditions.checkArgument(weights.shape().last() >= groupSize,
                "RMSNorm weights must cover groupSize columns");
    }

    private static double sumSquares(FloatBufferTensor tensor, int base, int groupSize) {
        FloatVector acc = FloatVector.zero(FLOAT_SPECIES);
        MemorySegment segment = tensor.getMemorySegment();
        long byteBase = (long) base * Float.BYTES;
        int upper = FLOAT_SPECIES.loopBound(groupSize);
        int i = 0;
        for (; i < upper; i += FLOAT_SPECIES.length()) {
            FloatVector v = FloatVector.fromMemorySegment(FLOAT_SPECIES, segment,
                    byteBase + (long) i * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            acc = acc.add(v.mul(v));
        }
        double sum = acc.reduceLanes(jdk.incubator.vector.VectorOperators.ADD);
        for (; i < groupSize; i++) {
            float value = segment.get(ValueLayout.JAVA_FLOAT, byteBase + (long) i * Float.BYTES);
            sum += value * value;
        }
        return sum;
    }

    private static void scaleGroup(FloatBufferTensor tensor, AbstractTensor weights, float[] decodedWeights, int base,
            int groupSize, float invRms) {
        FloatVector inv = FloatVector.broadcast(FLOAT_SPECIES, invRms);
        MemorySegment segment = tensor.getMemorySegment();
        long byteBase = (long) base * Float.BYTES;
        int upper = FLOAT_SPECIES.loopBound(groupSize);
        int i = 0;
        for (; i < upper; i += FLOAT_SPECIES.length()) {
            FloatVector values = FloatVector.fromMemorySegment(FLOAT_SPECIES, segment,
                    byteBase + (long) i * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
            FloatVector scaled = values.mul(inv);
            if (weights != null) {
                scaled = scaled.mul(weightVector(weights, decodedWeights, i));
            }
            scaled.intoMemorySegment(segment, byteBase + (long) i * Float.BYTES, ByteOrder.LITTLE_ENDIAN);
        }
        for (; i < groupSize; i++) {
            long byteOffset = byteBase + (long) i * Float.BYTES;
            float scaled = segment.get(ValueLayout.JAVA_FLOAT, byteOffset) * invRms;
            if (weights != null) {
                scaled *= decodedWeights == null ? weight(weights, i) : decodedWeights[i];
            }
            segment.set(ValueLayout.JAVA_FLOAT, byteOffset, scaled);
        }
    }

    private static FloatVector weightVector(AbstractTensor weights, float[] decodedWeights, int offset) {
        if (weights instanceof FloatBufferTensor floatWeights) {
            return floatWeights.getVector(FLOAT_SPECIES, 0, offset);
        }
        if (decodedWeights != null) {
            return FloatVector.fromArray(FLOAT_SPECIES, decodedWeights, offset);
        }
        throw new IllegalArgumentException("unsupported weights tensor " + weights.getClass());
    }

    private static float[] decodeBf16Weights(AbstractTensor weights, int groupSize) {
        BFloat16BufferTensor bf16Weights = (BFloat16BufferTensor) weights;
        float[] decoded = new float[groupSize];
        for (int i = 0; i < groupSize; i++) {
            short raw = bf16Weights.getMemorySegment().get(ValueLayout.JAVA_SHORT_UNALIGNED,
                    (long) i * Short.BYTES);
            decoded[i] = FloatConversions.bFloat16ToFloat32(raw);
        }
        return decoded;
    }

    private static float weight(AbstractTensor weights, int offset) {
        return weights.get(0, offset);
    }
}
