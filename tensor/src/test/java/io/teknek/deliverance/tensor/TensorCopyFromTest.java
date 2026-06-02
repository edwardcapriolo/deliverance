package io.teknek.deliverance.tensor;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.Float16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.function.IntFunction;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TensorCopyFromTest {

    @ParameterizedTest(name = "{0} copyFrom srcOffset={1} destOffset={2} length={3}")
    @MethodSource("denseTensorCopyWindows")
    public void denseCopyFromLengthAndOffsetsAreElementCounts(String name, int srcOffset, int destOffset, int length) {
        AbstractTensor src = denseTensor(name, 8);
        AbstractTensor dst = denseTensor(name, 8);
        fillRow(src, 1.0f);

        dst.copyFrom(src, srcOffset, destOffset, length);

        assertCopiedWindow(src, dst, srcOffset, destOffset, length, 0.0f);
    }

    @ParameterizedTest(name = "{0} copyFrom srcOffset={1} destOffset={2} length={3}")
    @MethodSource("quantizedTensorCopyWindows")
    public void quantizedCopyFromLengthAndOffsetsAreElementCounts(
            String name,
            IntFunction<AbstractTensor> factory,
            int srcOffset,
            int destOffset,
            int length
    ) {
        AbstractTensor src = factory.apply(1);
        AbstractTensor dst = factory.apply(-1);
        float[] before = rowValues(dst);

        dst.copyFrom(src, srcOffset, destOffset, length);

        assertCopiedWindow(src, dst, srcOffset, destOffset, length, 0.0001f);
        assertUnchangedOutsideWindow(dst, before, destOffset, length, 0.0001f);
    }

    @Test
    public void copyFromRejectsDifferentDTypes() {
        AbstractTensor src = new FloatBufferTensor(TensorShape.of(1, 8));
        AbstractTensor dst = new BFloat16BufferTensor(TensorShape.of(1, 8));

        assertThrows(IllegalArgumentException.class, () -> dst.copyFrom(src, 0, 0, 8));
    }

    private static Stream<Arguments> denseTensorCopyWindows() {
        return Stream.of("f32", "bf16", "f16")
                .flatMap(name -> copyWindows().map(window -> Arguments.of(
                        name,
                        window.srcOffset,
                        window.destOffset,
                        window.length
                )));
    }

    private static Stream<Arguments> quantizedTensorCopyWindows() {
        return Stream.of(
                Arguments.of("q8", (IntFunction<AbstractTensor>) sign -> quantizedTensor(DType.I8, sign), 0, 0, 32),
                Arguments.of("q8", (IntFunction<AbstractTensor>) sign -> quantizedTensor(DType.I8, sign), 4, 8, 8),
                Arguments.of("q4", (IntFunction<AbstractTensor>) sign -> quantizedTensor(DType.Q4, sign), 0, 0, 32),
                Arguments.of("q4", (IntFunction<AbstractTensor>) sign -> quantizedTensor(DType.Q4, sign), 4, 8, 8)
        );
    }

    private static Stream<CopyWindow> copyWindows() {
        return Stream.of(
                new CopyWindow(0, 0, 8),
                new CopyWindow(1, 2, 4),
                new CopyWindow(2, 0, 3),
                new CopyWindow(0, 3, 5)
        );
    }

    private static AbstractTensor denseTensor(String name, int columns) {
        TensorShape shape = TensorShape.of(1, columns);
        return switch (name) {
            case "f32" -> new FloatBufferTensor(shape);
            case "bf16" -> new BFloat16BufferTensor(shape);
            case "f16" -> new Float16BufferTensor(shape);
            default -> throw new IllegalArgumentException("Unknown tensor type " + name);
        };
    }

    private static AbstractTensor quantizedTensor(DType dtype, int sign) {
        FloatBufferTensor base = new FloatBufferTensor(TensorShape.of(1, 32));
        fillRow(base, sign);
        return AbstractTensorUtils.quantize(base, dtype);
    }

    private static void fillRow(AbstractTensor tensor, float sign) {
        for (int i = 0; i < tensor.shape().last(); i++) {
            tensor.set(sign * (i + 1.0f), 0, i);
        }
    }

    private static float[] rowValues(AbstractTensor tensor) {
        float[] values = new float[tensor.shape().last()];
        for (int i = 0; i < values.length; i++) {
            values[i] = tensor.get(0, i);
        }
        return values;
    }

    private static void assertCopiedWindow(AbstractTensor src, AbstractTensor dst, int srcOffset, int destOffset,
            int length, float delta) {
        for (int i = 0; i < length; i++) {
            assertEquals(src.get(0, srcOffset + i), dst.get(0, destOffset + i), delta,
                    "copied element at destination offset " + (destOffset + i));
        }
    }

    private static void assertUnchangedOutsideWindow(AbstractTensor dst, float[] before, int destOffset, int length,
            float delta) {
        for (int i = 0; i < before.length; i++) {
            if (i < destOffset || i >= destOffset + length) {
                assertEquals(before[i], dst.get(0, i), delta, "outside copied window at offset " + i);
            }
        }
    }

    private record CopyWindow(int srcOffset, int destOffset, int length) {
    }
}
