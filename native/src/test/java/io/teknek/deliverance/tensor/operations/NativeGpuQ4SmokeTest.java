package io.teknek.deliverance.tensor.operations;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

@Disabled("Manual WebGPU/Dawn smoke test. Requires local Dawn/WebGPU native libraries and a supported GPU backend.")
public class NativeGpuQ4SmokeTest {

    @ParameterizedTest(name = "f32 x q4 batch={0} rows={1} k={2}")
    @MethodSource("projectionCases")
    public void f32Q4ProjectionMatchesNaiveReference(int batchSize, int rows, int k) {
        try (FloatBufferTensor input = deterministicInput(batchSize, k);
             FloatBufferTensor denseWeight = deterministicWeight(rows, k);
             AbstractTensor q4Weight = AbstractTensorUtils.quantize(denseWeight, DType.Q4, true);
             FloatBufferTensor expected = new FloatBufferTensor(batchSize, rows);
             FloatBufferTensor actual = new FloatBufferTensor(batchSize, rows)) {
            NativeGPUTensorOperations gpu = new NativeGPUTensorOperations();
            gpu.registerModelTensor(q4Weight);

            new NaiveTensorOperations().batchDotProduct(expected, input, q4Weight, 0, 0, k, 0, 0, rows);
            gpu.batchDotProduct(actual, input, q4Weight, 0, 0, k, 0, 0, rows);

            assertTensorClose(expected, actual, 0.05f);
        }
    }

    private static Stream<Arguments> projectionCases() {
        return Stream.of(
                Arguments.of(1, 64, 128),
                Arguments.of(2, 128, 256),
                Arguments.of(1, 1024, 256)
        );
    }

    private static FloatBufferTensor deterministicInput(int rows, int cols) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(((row * 17 + col * 31) % 257 - 128) / 64.0f, row, col);
            }
        }
        return tensor;
    }

    private static FloatBufferTensor deterministicWeight(int rows, int cols) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(((row * 43 + col * 19) % 251 - 125) / 80.0f, row, col);
            }
        }
        return tensor;
    }

    private static void assertTensorClose(AbstractTensor expected, AbstractTensor actual, float tolerance) {
        assertEquals(expected.shape().first(), actual.shape().first());
        assertEquals(expected.shape().last(), actual.shape().last());
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int col = 0; col < expected.shape().last(); col++) {
                assertEquals(expected.get(row, col), actual.get(row, col), tolerance,
                        "row=" + row + " col=" + col);
            }
        }
    }
}
