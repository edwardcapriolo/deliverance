package io.teknek.deliverance.tensor.operations;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Optional;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NativeSimdGemma4Q4ParityTest {

    @ParameterizedTest(name = "{0} {1} batch={2} rows={3} k={4} bRowOffset={5} rowChunk={6}")
    @MethodSource("providerGemma4ProjectionCases")
    public void gemma4SizedQ4ProjectionMatchesNaiveReference(String providerName, TensorOperations ops,
            String name, int batchSize, int rows, int k,
            int bRowOffset, int rowChunk, DType activationType, float tolerance) {
        try (FloatBufferTensor denseInput = deterministicInput(batchSize, k);
             FloatBufferTensor denseWeight = deterministicWeight(rows, k);
             AbstractTensor activation = quantizeActivation(denseInput, activationType);
              AbstractTensor q4Weight = AbstractTensorUtils.quantize(denseWeight, DType.Q4, true);
              FloatBufferTensor expected = new FloatBufferTensor(batchSize, rows);
              FloatBufferTensor actual = new FloatBufferTensor(batchSize, rows)) {
            new NaiveTensorOperations().batchDotProduct(expected, activation, q4Weight, 0, 0, k, 0, bRowOffset, rowChunk);
            ops.registerModelTensor(q4Weight);
            ops.batchDotProduct(actual, activation, q4Weight, 0, 0, k, 0, bRowOffset, rowChunk);

            assertTensorClose(expected, actual, tolerance, providerName + " " + name);
        }
    }

    @ParameterizedTest(name = "batched {0} {1} batch={2} rows={3} k={4}")
    @MethodSource("providerGemma4BatchedProjectionCases")
    public void gemma4SizedBatchedQ4ProjectionMatchesSingleProjection(String providerName, TensorOperations ops,
            String name, int batchSize, int rows, int k,
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
            ops.batchDotProduct(expectedGate, activation, q4GateWeight, 0, 0, k, 0, 0, rows);
            ops.batchDotProduct(expectedUp, activation, q4UpWeight, 0, 0, k, 0, 0, rows);
            ops.registerModelTensor(q4GateWeight);
            ops.registerModelTensor(q4UpWeight);
            ops.dotProductBatchChunk(new AbstractTensor[]{actualGate, actualUp}, activation,
                    new AbstractTensor[]{q4GateWeight, q4UpWeight}, 0, k, 0, rows);

            assertTensorClose(expectedGate, actualGate, tolerance, providerName + " " + name + " gate");
            assertTensorClose(expectedUp, actualUp, tolerance, providerName + " " + name + " up");
        }
    }

    private static Stream<Arguments> providerGemma4ProjectionCases() {
        return providers().flatMap(provider -> Stream.of(
                Arguments.of("sliding-q", 3, 2_048, 1_536, 0, 2_048, DType.BF16, 0.03f),
                Arguments.of("full-q-offset", 2, 4_096, 1_536, 511, 513, DType.BF16, 0.03f),
                Arguments.of("mlp-i8-q4-offset", 2, 6_144, 1_536, 2_047, 1_025, DType.I8, 0.08f)
        ).map(testCase -> prependProvider(provider, testCase)));
    }

    private static Stream<Arguments> providerGemma4BatchedProjectionCases() {
        return providers().map(provider -> prependProvider(provider,
                Arguments.of("mlp-gate-up-i8-q4", 2, 6_144, 1_536, DType.I8, 0.08f)));
    }

    private static Stream<Arguments> providers() {
        WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        ArrayQueueTensorAllocator allocator = new ArrayQueueTensorAllocator(new MetricRegistry());
        TensorOperations naive = new NaiveTensorOperations();
        TensorOperations panama = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool);
        return Stream.of(
                Optional.of(Arguments.of("naive", naive)),
                Optional.of(Arguments.of("panama", panama)),
                simd(panama).map(ops -> Arguments.of("simd", ops)),
                gpu().map(ops -> Arguments.of("gpu", ops))
        ).flatMap(Optional::stream);
    }

    private static Optional<TensorOperations> simd(TensorOperations fallback) {
        try {
            return Optional.of(new NativeSimdTensorOperations(fallback));
        } catch (Throwable t) {
            return Optional.empty();
        }
    }

    private static Optional<TensorOperations> gpu() {
        try {
            return Optional.of(new NativeGPUTensorOperations());
        } catch (Throwable t) {
            return Optional.empty();
        }
    }

    private static Arguments prependProvider(Arguments provider, Arguments testCase) {
        Object[] providerArgs = provider.get();
        Object[] caseArgs = testCase.get();
        Object[] combined = new Object[providerArgs.length + caseArgs.length];
        System.arraycopy(providerArgs, 0, combined, 0, providerArgs.length);
        System.arraycopy(caseArgs, 0, combined, providerArgs.length, caseArgs.length);
        return Arguments.of(combined);
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
