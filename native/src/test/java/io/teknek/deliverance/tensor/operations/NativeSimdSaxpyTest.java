package io.teknek.deliverance.tensor.operations;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NativeSimdSaxpyTest {

    @ParameterizedTest(name = "scalar xoffset={0} yoffset={1} limit={2}")
    @MethodSource("scalarCases")
    public void scalarF32SaxpyMatchesNaiveReference(int xoffset, int yoffset, int limit) {
        try (FloatBufferTensor x = vector(1, xoffset + limit + 3);
              FloatBufferTensor expected = vector(1, yoffset + limit + 3);
              FloatBufferTensor actual = copy(expected);
              FloatBufferTensor panama = copy(expected)) {
            new NaiveTensorOperations().saxpy(1.75f, x, expected, xoffset, yoffset, limit);
            new NativeSimdTensorOperations(new NaiveTensorOperations()).saxpy(1.75f, x, actual, xoffset, yoffset, limit);

            assertTensorClose(expected, actual);
            panamaOps().saxpy(1.75f, x, panama, xoffset, yoffset, limit);
            assertTensorClose(expected, panama);
        }
    }

    @ParameterizedTest(name = "batch xoffset={0} yoffset={1} limit={2} aOffset={3} xRowOffset={4} batch={5}")
    @MethodSource("batchCases")
    public void batchedF32SaxpyMatchesNaiveReference(int xoffset, int yoffset, int limit, int aOffset,
            int xRowOffset, int batchSize) {
        try (FloatBufferTensor alpha = vector(1, aOffset + batchSize + 3);
              FloatBufferTensor x = vector(xRowOffset + batchSize + 2, xoffset + limit + 3);
              FloatBufferTensor expected = vector(1, yoffset + limit + 3);
              FloatBufferTensor actual = copy(expected);
              FloatBufferTensor panama = copy(expected)) {
            new NaiveTensorOperations().saxpy(alpha, x, expected, xoffset, yoffset, limit, aOffset, xRowOffset, batchSize);
            new NativeSimdTensorOperations(new NaiveTensorOperations()).saxpy(alpha, x, actual,
                    xoffset, yoffset, limit, aOffset, xRowOffset, batchSize);

            assertTensorClose(expected, actual);
            panamaOps().saxpy(alpha, x, panama, xoffset, yoffset, limit, aOffset, xRowOffset, batchSize);
            assertTensorClose(expected, panama);
        }
    }

    private static Stream<Arguments> scalarCases() {
        return Stream.of(
                Arguments.of(0, 0, 32),
                Arguments.of(3, 5, 31),
                Arguments.of(2, 1, 9)
        );
    }

    private static Stream<Arguments> batchCases() {
        return Stream.of(
                Arguments.of(0, 0, 32, 0, 0, 4),
                Arguments.of(2, 3, 31, 1, 2, 5),
                Arguments.of(1, 4, 9, 3, 1, 7),
                Arguments.of(0, 2, 128, 0, 0, 32)
        );
    }

    private static FloatBufferTensor vector(int rows, int cols) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(((row * 17 + col * 31) % 257 - 128) / 64.0f, row, col);
            }
        }
        return tensor;
    }

    private static FloatBufferTensor copy(AbstractTensor source) {
        FloatBufferTensor copy = new FloatBufferTensor(source.shape());
        for (int row = 0; row < source.shape().first(); row++) {
            for (int col = 0; col < source.shape().last(); col++) {
                copy.set(source.get(row, col), row, col);
            }
        }
        return copy;
    }

    private static PanamaTensorOperations panamaOps() {
        return new PanamaTensorOperations(
                MachineSpec.VECTOR_TYPE,
                new ArrayQueueTensorAllocator(new MetricRegistry()),
                new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())
        );
    }

    private static void assertTensorClose(AbstractTensor expected, AbstractTensor actual) {
        assertEquals(expected.shape().first(), actual.shape().first());
        assertEquals(expected.shape().last(), actual.shape().last());
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int col = 0; col < expected.shape().last(); col++) {
                assertEquals(expected.get(row, col), actual.get(row, col), 0.001f,
                        "row=" + row + " col=" + col);
            }
        }
    }
}
