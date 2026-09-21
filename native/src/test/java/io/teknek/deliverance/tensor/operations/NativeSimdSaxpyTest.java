package io.teknek.deliverance.tensor.operations;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NativeSimdSaxpyTest {

    @ParameterizedTest(name = "scalar {0}")
    @MethodSource("scalarCases")
    public void scalarF32SaxpyMatchesNaiveReference(SaxpyCases.ScalarCase c) {
        try (FloatBufferTensor x = vector(1, c.xOffset() + c.length() + 3);
              FloatBufferTensor expected = vector(1, c.yOffset() + c.length() + 3);
              FloatBufferTensor actual = copy(expected);
              FloatBufferTensor panama = copy(expected)) {
            new NaiveTensorOperations().saxpy(1.75f, x, expected, c.xOffset(), c.yOffset(), c.length());
            new NativeSimdTensorOperations(new NaiveTensorOperations()).saxpy(1.75f, x, actual,
                    c.xOffset(), c.yOffset(), c.length());

            assertTensorClose(expected, actual);
            panamaOps().saxpy(1.75f, x, panama, c.xOffset(), c.yOffset(), c.length());
            assertTensorClose(expected, panama);
        }
    }

    @ParameterizedTest(name = "batch {0}")
    @MethodSource("batchCases")
    public void batchedF32SaxpyMatchesNaiveReference(SaxpyCases.BatchCase c) {
        try (FloatBufferTensor alpha = vector(1, c.alphaOffset() + c.batchSize() + 3);
              FloatBufferTensor x = vector(c.xRowOffset() + c.batchSize() + 2, c.xOffset() + c.length() + 3);
              FloatBufferTensor expected = vector(1, c.yOffset() + c.length() + 3);
              FloatBufferTensor actual = copy(expected);
              FloatBufferTensor panama = copy(expected)) {
            new NaiveTensorOperations().saxpy(alpha, x, expected, c.xOffset(), c.yOffset(), c.length(),
                    c.alphaOffset(), c.xRowOffset(), c.batchSize());
            new NativeSimdTensorOperations(new NaiveTensorOperations()).saxpy(alpha, x, actual,
                    c.xOffset(), c.yOffset(), c.length(), c.alphaOffset(), c.xRowOffset(), c.batchSize());

            assertTensorClose(expected, actual);
            panamaOps().saxpy(alpha, x, panama, c.xOffset(), c.yOffset(), c.length(), c.alphaOffset(),
                    c.xRowOffset(), c.batchSize());
            assertTensorClose(expected, panama);
        }
    }

    @Test
    public void batchedI8F32SaxpyMatchesPanamaReference() {
        try (FloatBufferTensor alpha = vector(1, 4);
             FloatBufferTensor denseX = vector(4, 64);
             AbstractTensor x = AbstractTensorUtils.quantize(denseX, DType.I8, true);
             FloatBufferTensor expected = vector(1, 64);
             FloatBufferTensor actual = copy(expected)) {
            panamaOps().saxpy(alpha, x, expected, 0, 0, 64, 0, 0, 4);
            new NativeSimdTensorOperations(panamaOps()).saxpy(alpha, x, actual, 0, 0, 64, 0, 0, 4);

            assertTensorClose(expected, actual);
        }
    }

    private static Stream<Arguments> scalarCases() {
        return SaxpyCases.scalarCases();
    }

    private static Stream<Arguments> batchCases() {
        return SaxpyCases.batchCases();
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
