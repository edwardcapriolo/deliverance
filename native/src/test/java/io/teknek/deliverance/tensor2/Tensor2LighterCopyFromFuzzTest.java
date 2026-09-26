package io.teknek.deliverance.tensor2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import org.junit.jupiter.api.Test;

import java.lang.foreign.ValueLayout;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Tensor2LighterCopyFromFuzzTest {
    @Test
    void f32CopyFromMatchesLogicalOffsetContract() {
        Lighter lighter = panamaOnly();
        try (TensorRef source = lighter.allocate(DType.F32, TensorShape.of(1, 128));
             TensorRef target = lighter.allocate(DType.F32, TensorShape.of(1, 128))) {
            fill(source, 11);
            fill(target, 23);
            float[] expected = row(target);
            for (int column = 0; column < 47; column++) {
                expected[61 + column] = source.get(0, 13 + column);
            }
            lighter.copyFrom(source, target, 13, 61, 47);
            assertRow(target, expected, 0.0f);
        }
    }

    @Test
    void q4CopyFromMatchesLegacyPayloadAndScalesForPartialBlocks() {
        Lighter lighter = panamaOnly();
        for (int[] copy : new int[][] {{0, 64, 64}, {16, 48, 48}, {31, 33, 33}}) {
            int sourceOffset = copy[0];
            int targetOffset = copy[1];
            int length = copy[2];
            try (FloatBufferTensor sourceDense = dense(128, 31);
                 FloatBufferTensor targetDense = dense(128, 47);
                 Q4ByteBufferTensor legacySource = new Q4ByteBufferTensor(sourceDense);
                 Q4ByteBufferTensor legacyTarget = new Q4ByteBufferTensor(targetDense);
                 TensorRef sourceF32 = lighter.allocate(DType.F32, TensorShape.of(1, 128));
                 TensorRef targetF32 = lighter.allocate(DType.F32, TensorShape.of(1, 128))) {
                copy(sourceDense, sourceF32);
                copy(targetDense, targetF32);
                try (TensorRef source = lighter.reshape(sourceF32, DType.Q4);
                     TensorRef target = lighter.reshape(targetF32, DType.Q4)) {
                    legacyTarget.copyFrom(legacySource, sourceOffset, targetOffset, length);
                    lighter.copyFrom(source, target, sourceOffset, targetOffset, length);
                    Q4Tensor actual = (Q4Tensor) target.underlying();
                    int payloadBytes = length / 2;
                    for (int byteIndex = 0; byteIndex < payloadBytes; byteIndex++) {
                        int sourceByte = sourceOffset / 2 + byteIndex;
                        int targetByte = targetOffset / 2 + byteIndex;
                        assertEquals(legacyTarget.getMemorySegment().get(ValueLayout.JAVA_BYTE, targetByte),
                                actual.getRawPackedByte(targetByte), "payload=" + java.util.Arrays.toString(copy)
                                        + " byte=" + byteIndex + " source=" + sourceByte);
                    }
                    for (int block = 0; block < 4; block++) {
                        assertEquals(legacyTarget.getBlockF().get(0, block), Q4Layout.scale(target).get(0, block),
                                0.0f, "scale=" + java.util.Arrays.toString(copy) + " block=" + block);
                    }
                }
            }
        }
    }

    private static Lighter panamaOnly() {
        return new Lighter(new MetricRegistry(), Map.of(TensorProviderKind.PANAMA, new PanamaOps()));
    }

    private static FloatBufferTensor dense(int columns, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(TensorShape.of(1, columns));
        fill(tensor, seed);
        return tensor;
    }

    private static void fill(FloatBufferTensor target, int seed) {
        for (int column = 0; column < target.shape().last(); column++) {
            target.set(((column * 17 + seed) % 61 - 30) / 8.0f, 0, column);
        }
    }

    private static void fill(TensorRef target, int seed) {
        for (int column = 0; column < target.shape().last(); column++) {
            target.set(((column * 17 + seed) % 61 - 30) / 8.0f, 0, column);
        }
    }

    private static void copy(FloatBufferTensor source, TensorRef target) {
        for (int column = 0; column < source.shape().last(); column++) {
            target.set(source.get(0, column), 0, column);
        }
    }

    private static float[] row(TensorRef source) {
        float[] result = new float[source.shape().last()];
        for (int column = 0; column < result.length; column++) {
            result[column] = source.get(0, column);
        }
        return result;
    }

    private static void assertRow(TensorRef actual, float[] expected, float tolerance) {
        for (int column = 0; column < expected.length; column++) {
            assertEquals(expected[column], actual.get(0, column), tolerance, "column=" + column);
        }
    }
}
