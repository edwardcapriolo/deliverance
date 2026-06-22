package io.teknek.deliverance.tensor.operations;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NativeSimdGemma4Q4ParityTest {

    @ParameterizedTest(name = "{0} batch={1} rows={2} k={3} bRowOffset={4} rowChunk={5}")
    @MethodSource("gemma4ProjectionCases")
    public void gemma4SizedQ4ProjectionMatchesNaiveReference(String name, int batchSize, int rows, int k,
            int bRowOffset, int rowChunk, DType activationType, float tolerance) {
        try (FloatBufferTensor denseInput = deterministicInput(batchSize, k);
             FloatBufferTensor denseWeight = deterministicWeight(rows, k);
             AbstractTensor activation = quantizeActivation(denseInput, activationType);
             AbstractTensor q4Weight = AbstractTensorUtils.quantize(denseWeight, DType.Q4, true);
             FloatBufferTensor expected = new FloatBufferTensor(batchSize, rows);
             FloatBufferTensor actual = new FloatBufferTensor(batchSize, rows)) {
            new NaiveTensorOperations().batchDotProduct(expected, activation, q4Weight, 0, 0, k, 0, bRowOffset, rowChunk);
            new NativeSimdTensorOperations(new NaiveTensorOperations()).batchDotProduct(actual, activation, q4Weight,
                    0, 0, k, 0, bRowOffset, rowChunk);

            assertTensorClose(expected, actual, tolerance, name);
        }
    }

    @ParameterizedTest(name = "batched {0} batch={1} rows={2} k={3}")
    @MethodSource("gemma4BatchedProjectionCases")
    public void gemma4SizedBatchedQ4ProjectionMatchesSingleProjection(String name, int batchSize, int rows, int k,
            DType activationType, float tolerance) {
        try (FloatBufferTensor denseInput = deterministicInput(batchSize, k);
             FloatBufferTensor denseGateWeight = deterministicWeight(rows, k);
             FloatBufferTensor denseUpWeight = deterministicWeightVariant(rows, k);
             AbstractTensor activation = quantizeActivation(denseInput, activationType);
             AbstractTensor q4GateWeight = AbstractTensorUtils.quantize(denseGateWeight, DType.Q4, true);
             AbstractTensor q4UpWeight = AbstractTensorUtils.quantize(denseUpWeight, DType.Q4, true);
             FloatBufferTensor expectedGate = new FloatBufferTensor(batchSize, rows);
             FloatBufferTensor expectedUp = new FloatBufferTensor(batchSize, rows);
             FloatBufferTensor actualGate = new FloatBufferTensor(batchSize, rows);
             FloatBufferTensor actualUp = new FloatBufferTensor(batchSize, rows)) {
            NativeSimdTensorOperations ops = new NativeSimdTensorOperations(new NaiveTensorOperations());
            ops.batchDotProduct(expectedGate, activation, q4GateWeight, 0, 0, k, 0, 0, rows);
            ops.batchDotProduct(expectedUp, activation, q4UpWeight, 0, 0, k, 0, 0, rows);
            ops.dotProductBatchChunk(new AbstractTensor[]{actualGate, actualUp}, activation,
                    new AbstractTensor[]{q4GateWeight, q4UpWeight}, 0, k, 0, rows);

            assertTensorClose(expectedGate, actualGate, tolerance, name + " gate");
            assertTensorClose(expectedUp, actualUp, tolerance, name + " up");
        }
    }

    private static Stream<Arguments> gemma4ProjectionCases() {
        return Stream.of(
                Arguments.of("sliding-q", 3, 2_048, 1_536, 0, 2_048, DType.BF16, 0.03f),
                Arguments.of("full-q-offset", 2, 4_096, 1_536, 511, 513, DType.BF16, 0.03f),
                Arguments.of("mlp-i8-q4-offset", 2, 6_144, 1_536, 2_047, 1_025, DType.I8, 0.08f)
        );
    }

    private static Stream<Arguments> gemma4BatchedProjectionCases() {
        return Stream.of(
                Arguments.of("mlp-gate-up-i8-q4", 2, 6_144, 1_536, DType.I8, 0.08f)
        );
    }

    private static AbstractTensor quantizeActivation(AbstractTensor input, DType activationType) {
        if (activationType == DType.BF16) {
            return new BFloat16BufferTensor(input);
        }
        return AbstractTensorUtils.quantize(input, activationType, true);
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

    private static FloatBufferTensor deterministicWeightVariant(int rows, int cols) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(((row * 29 + col * 37) % 241 - 120) / 72.0f, row, col);
            }
        }
        return tensor;
    }

    private static void assertTensorClose(AbstractTensor expected, AbstractTensor actual, float tolerance, String label) {
        assertEquals(expected.shape().first(), actual.shape().first());
        assertEquals(expected.shape().last(), actual.shape().last());
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int col = 0; col < expected.shape().last(); col++) {
                assertEquals(expected.get(row, col), actual.get(row, col), tolerance,
                        label + " row=" + row + " col=" + col);
            }
        }
    }
}
