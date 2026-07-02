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

public class NativeGpuGemmParityTest {

    @ParameterizedTest(name = "{0} {1} batch={2} rows={3} k={4}")
    @MethodSource("providerGemmCases")
    public void gpuSupportedGemmPathsMatchNaiveReference(String providerName, TensorOperations ops,
            String name, int batchSize, int rows, int k, DType inputType, DType weightType, float tolerance) {
        try (FloatBufferTensor denseInput = deterministicInput(batchSize, k);
             FloatBufferTensor denseWeight = deterministicWeight(rows, k);
             AbstractTensor input = convertInput(denseInput, inputType);
             AbstractTensor weight = convertWeight(denseWeight, weightType);
             FloatBufferTensor expected = new FloatBufferTensor(batchSize, rows);
             FloatBufferTensor actual = new FloatBufferTensor(batchSize, rows)) {

            new NaiveTensorOperations().batchDotProduct(expected, input, weight, 0, 0, k, 0, 0, rows);
            ops.registerModelTensor(weight);
            ops.batchDotProduct(actual, input, weight, 0, 0, k, 0, 0, rows);

            assertTensorClose(expected, actual, tolerance, providerName + " " + name);
        }
    }

    private static Stream<Arguments> providerGemmCases() {
        return providers().flatMap(provider -> Stream.of(
                Arguments.of("f32xf32", 3, 1_024, 128, DType.F32, DType.F32, 0.0001f),
                Arguments.of("f32xbf16", 3, 1_024, 128, DType.F32, DType.BF16, 0.04f),
                Arguments.of("f32xq4", 3, 1_024, 128, DType.F32, DType.Q4, 0.08f),
                Arguments.of("i8xq4-m1", 1, 1_024, 128, DType.I8, DType.Q4, 0.20f),
                Arguments.of("i8xq4", 3, 1_024, 128, DType.I8, DType.Q4, 0.20f)
        ).map(testCase -> prependProvider(provider, testCase)));
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

    private static AbstractTensor convertInput(AbstractTensor input, DType inputType) {
        if (inputType == DType.F32) {
            return new FloatBufferTensor(input);
        }
        return AbstractTensorUtils.quantize(input, inputType, true);
    }

    private static AbstractTensor convertWeight(AbstractTensor weight, DType weightType) {
        if (weightType == DType.F32) {
            return new FloatBufferTensor(weight);
        }
        if (weightType == DType.BF16) {
            return new BFloat16BufferTensor(weight);
        }
        return AbstractTensorUtils.quantize(weight, weightType, true);
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

    private static void assertTensorClose(AbstractTensor expected, AbstractTensor actual, float tolerance, String label) {
        assertEquals(expected.shape(), actual.shape(), label + " shape");
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int col = 0; col < expected.shape().last(); col++) {
                assertEquals(expected.get(row, col), actual.get(row, col), tolerance,
                        label + " row=" + row + " col=" + col);
            }
        }
    }
}
