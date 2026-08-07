package io.teknek.deliverance.tensor.operations;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.concurrent.ForkJoinPool;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

class xTensorOperationsSlicePrimitiveTest {

    @ParameterizedTest(name = "{0} dotSlice")
    @MethodSource("operations")
    void dotSliceMatchesReference(String name, TensorOperations ops) {
        try (AbstractTensor left = tensor(2, 11, 3);
             AbstractTensor right = tensor(3, 13, 7)) {
            float expected = 0.0f;
            for (int i = 0; i < 7; i++) {
                expected += left.get(1, 2 + i) * right.get(2, 4 + i);
            }

            float actual = ops.dotSlice(left, 1, 2, right, 2, 4, 7);

            assertEquals(expected, actual, 1.0e-6f, name);
        }
    }

    @ParameterizedTest(name = "{0} dotRowsToArray")
    @MethodSource("operations")
    void dotRowsToArrayMatchesReference(String name, TensorOperations ops) {
        try (AbstractTensor left = tensor(1, 11, 3);
             AbstractTensor rows = tensor(4, 13, 7)) {
            float[] expected = new float[3];
            for (int row = 0; row < expected.length; row++) {
                for (int i = 0; i < 7; i++) {
                    expected[row] += left.get(0, 2 + i) * rows.get(1 + row, 4 + i);
                }
            }
            float[] actual = new float[3];

            ops.dotRowsToArray(left, 0, 2, rows, 1, 4, 3, 7, actual, 0);

            for (int i = 0; i < expected.length; i++) {
                assertEquals(expected[i], actual[i], 1.0e-6f, name + " row=" + i);
            }
        }
    }

    @ParameterizedTest(name = "{0} weightedRescaleAccumulateSlice")
    @MethodSource("operations")
    void weightedRescaleAccumulateSliceMatchesReference(String name, TensorOperations ops) {
        try (AbstractTensor out = tensor(1, 12, 3);
             AbstractTensor value = tensor(3, 14, 11)) {
            float[] expected = slice(out, 3, 6);
            for (int i = 0; i < expected.length; i++) {
                expected[i] = expected[i] * 0.25f + value.get(2, 5 + i) * 0.75f;
            }

            ops.weightedRescaleAccumulateSlice(out, 0, 3, value, 2, 5, 6, 0.25f, 0.75f);

            assertSliceEquals(expected, out, 3, name);
        }
    }

    @ParameterizedTest(name = "{0} accumulateWeightedSlice")
    @MethodSource("operations")
    void accumulateWeightedSliceMatchesReference(String name, TensorOperations ops) {
        try (AbstractTensor out = tensor(1, 12, 5);
             AbstractTensor value = tensor(3, 14, 13)) {
            float[] expected = slice(out, 4, 5);
            for (int i = 0; i < expected.length; i++) {
                expected[i] += value.get(1, 6 + i) * -0.5f;
            }

            ops.accumulateWeightedSlice(out, 0, 4, value, 1, 6, 5, -0.5f);

            assertSliceEquals(expected, out, 4, name);
        }
    }

    @ParameterizedTest(name = "{0} accumulateWeightedRows")
    @MethodSource("operations")
    void accumulateWeightedRowsMatchesReference(String name, TensorOperations ops) {
        try (AbstractTensor out = tensor(1, 12, 5);
             AbstractTensor rows = tensor(4, 14, 13)) {
            float[] weights = { 0.5f, -0.25f, 0.125f };
            float[] expected = slice(out, 4, 5);
            for (int row = 0; row < weights.length; row++) {
                for (int i = 0; i < expected.length; i++) {
                    expected[i] += rows.get(1 + row, 6 + i) * weights[row];
                }
            }

            ops.accumulateWeightedRows(out, 0, 4, rows, 1, 6, 3, 5, weights, 0);

            assertSliceEquals(expected, out, 4, name);
        }
    }

    @ParameterizedTest(name = "{0} normalizeSlice")
    @MethodSource("operations")
    void normalizeSliceMatchesReference(String name, TensorOperations ops) {
        try (AbstractTensor out = tensor(2, 12, 17)) {
            float[] expected = slice(out, 1, 2, 8);
            for (int i = 0; i < expected.length; i++) {
                expected[i] *= 0.125f;
            }

            ops.normalizeSlice(out, 1, 2, 8, 0.125f);

            for (int i = 0; i < expected.length; i++) {
                assertEquals(expected[i], out.get(1, 2 + i), 1.0e-6f, name + " col=" + i);
            }
        }
    }

    @ParameterizedTest(name = "{0} scaleSlice")
    @MethodSource("operations")
    void scaleSliceMatchesReference(String name, TensorOperations ops) {
        try (AbstractTensor out = tensor(2, 12, 19)) {
            float[] expected = slice(out, 1, 3, 6);
            for (int i = 0; i < expected.length; i++) {
                expected[i] *= -0.25f;
            }

            ops.scaleSlice(out, 1, 3, 6, -0.25f);

            for (int i = 0; i < expected.length; i++) {
                assertEquals(expected[i], out.get(1, 3 + i), 1.0e-6f, name + " col=" + i);
            }
        }
    }

    private static Stream<Arguments> operations() {
        WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(2));
        return Stream.of(
                Arguments.of("naive", new NaiveTensorOperations()),
                Arguments.of("panama", new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                        new ArrayQueueTensorAllocator(new MetricRegistry()), pool))
        );
    }

    private static AbstractTensor tensor(int rows, int cols, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set((((row * 19 + col * 23 + seed) % 43) - 21) / 21.0f, row, col);
            }
        }
        return tensor;
    }

    private static float[] slice(AbstractTensor tensor, int offset, int length) {
        return slice(tensor, 0, offset, length);
    }

    private static float[] slice(AbstractTensor tensor, int row, int offset, int length) {
        float[] values = new float[length];
        for (int i = 0; i < length; i++) {
            values[i] = tensor.get(row, offset + i);
        }
        return values;
    }

    private static void assertSliceEquals(float[] expected, AbstractTensor actual, int offset, String name) {
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual.get(0, offset + i), 1.0e-6f, name + " col=" + i);
        }
    }
}
