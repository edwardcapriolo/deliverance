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

    @ParameterizedTest(name = "{0} offset {1} batch={2} rows={3} k={4} aOffset={5} bOffset={6} rOffset={7} bRowOffset={8} chunk={9}")
    @MethodSource("providerOffsetGemmCases")
    public void gemmOffsetAndTailPathsMatchPanamaBaseline(String providerName, TensorOperations ops,
            String name, int batchSize, int rows, int k, int aColumnOffset, int bColumnOffset,
            int rRowOffset, int bRowOffset, int rowChunkSize, DType inputType, DType weightType, float tolerance) {
        int inputCols = alignToBlock(aColumnOffset + k);
        int weightCols = alignToBlock(bColumnOffset + k);
        int resultCols = bRowOffset + rowChunkSize + rRowOffset;
        TensorOperations panama = panama();
        try (FloatBufferTensor denseInput = deterministicInput(batchSize, inputCols);
             FloatBufferTensor denseWeight = deterministicWeight(rows, weightCols);
             AbstractTensor input = convertInput(denseInput, inputType);
             AbstractTensor weight = convertWeight(denseWeight, weightType);
             FloatBufferTensor reference = new FloatBufferTensor(batchSize, resultCols);
             FloatBufferTensor expected = new FloatBufferTensor(batchSize, resultCols);
             FloatBufferTensor actual = new FloatBufferTensor(batchSize, resultCols)) {

            new NaiveTensorOperations().batchDotProduct(reference, input, weight, aColumnOffset, bColumnOffset, k,
                    rRowOffset, bRowOffset, rowChunkSize);
            panama.registerModelTensor(weight);
            panama.batchDotProduct(expected, input, weight, aColumnOffset, bColumnOffset, k,
                    rRowOffset, bRowOffset, rowChunkSize);
            assertTensorClose(reference, expected, tolerance, "panama baseline " + name);

            ops.registerModelTensor(weight);
            ops.batchDotProduct(actual, input, weight, aColumnOffset, bColumnOffset, k,
                    rRowOffset, bRowOffset, rowChunkSize);
            assertTensorClose(expected, actual, tolerance, providerName + " " + name);
        }
    }

    @ParameterizedTest(name = "{0} batchChunk {1} batch={2} rows={3} k={4} offset={5} chunkStart={6} chunk={7}")
    @MethodSource("providerBatchChunkCases")
    public void dotProductBatchChunkOffsetAndTailPathsMatchPanamaBaseline(String providerName, TensorOperations ops,
            String name, int batchSize, int rows, int k, int columnOffset, int chunkStart, int chunkSize,
            DType inputType, DType weightType, float tolerance) {
        int cols = alignToBlock(columnOffset + k);
        int resultCols = chunkStart + chunkSize;
        TensorOperations panama = panama();
        try (FloatBufferTensor denseInput = deterministicInput(batchSize, cols);
             FloatBufferTensor denseWeight0 = deterministicWeight(rows, cols);
             FloatBufferTensor denseWeight1 = deterministicWeightVariant(rows, cols);
             AbstractTensor input = convertInput(denseInput, inputType);
             AbstractTensor weight0 = convertWeight(denseWeight0, weightType);
             AbstractTensor weight1 = convertWeight(denseWeight1, weightType);
             FloatBufferTensor reference0 = new FloatBufferTensor(batchSize, resultCols);
             FloatBufferTensor reference1 = new FloatBufferTensor(batchSize, resultCols);
             FloatBufferTensor expected0 = new FloatBufferTensor(batchSize, resultCols);
             FloatBufferTensor expected1 = new FloatBufferTensor(batchSize, resultCols);
             FloatBufferTensor actual0 = new FloatBufferTensor(batchSize, resultCols);
             FloatBufferTensor actual1 = new FloatBufferTensor(batchSize, resultCols)) {

            new NaiveTensorOperations().batchDotProduct(reference0, input, weight0, columnOffset, columnOffset, k,
                    0, chunkStart, chunkSize);
            new NaiveTensorOperations().batchDotProduct(reference1, input, weight1, columnOffset, columnOffset, k,
                    0, chunkStart, chunkSize);

            panama.registerModelTensor(weight0);
            panama.registerModelTensor(weight1);
            panama.dotProductBatchChunk(new AbstractTensor[]{expected0, expected1}, input,
                    new AbstractTensor[]{weight0, weight1}, columnOffset, k, chunkStart, chunkSize);
            assertTensorClose(reference0, expected0, tolerance, "panama batchChunk " + name + " weight0");
            assertTensorClose(reference1, expected1, tolerance, "panama batchChunk " + name + " weight1");

            ops.registerModelTensor(weight0);
            ops.registerModelTensor(weight1);
            ops.dotProductBatchChunk(new AbstractTensor[]{actual0, actual1}, input,
                    new AbstractTensor[]{weight0, weight1}, columnOffset, k, chunkStart, chunkSize);
            assertTensorClose(expected0, actual0, tolerance, providerName + " batchChunk " + name + " weight0");
            assertTensorClose(expected1, actual1, tolerance, providerName + " batchChunk " + name + " weight1");
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

    private static Stream<Arguments> providerOffsetGemmCases() {
        return providers().flatMap(provider -> Stream.of(
                Arguments.of("f32xf32-offset-tail", 3, 37, 127, 3, 5, 11, 7, 19, DType.F32, DType.F32, 0.0001f),
                Arguments.of("f32xbf16-offset-tail", 2, 41, 95, 1, 9, 17, 13, 17, DType.F32, DType.BF16, 0.04f),
                Arguments.of("f32xq4-offset-tail", 3, 43, 127, 5, 3, 12, 9, 23, DType.F32, DType.Q4, 0.08f),
                Arguments.of("i8xq4-offset-tail", 2, 39, 96, 7, 11, 14, 8, 21, DType.I8, DType.Q4, 0.20f)
        ).map(testCase -> prependProvider(provider, testCase)));
    }

    private static Stream<Arguments> providerBatchChunkCases() {
        return providers().flatMap(provider -> Stream.of(
                Arguments.of("f32xf32-batchchunk-offset-tail", 3, 47, 127, 3, 11, 19, DType.F32, DType.F32, 0.0001f),
                Arguments.of("f32xq4-batchchunk-offset-tail", 3, 53, 127, 5, 9, 23, DType.F32, DType.Q4, 0.08f),
                Arguments.of("i8xq4-batchchunk-offset-tail", 2, 43, 96, 7, 8, 21, DType.I8, DType.Q4, 0.20f)
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

    private static TensorOperations panama() {
        return new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                new ArrayQueueTensorAllocator(new MetricRegistry()),
                new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores()));
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

    private static int alignToBlock(int value) {
        int block = 32;
        return ((value + block - 1) / block) * block;
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
        assertEquals(expected.shape(), actual.shape(), label + " shape");
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int col = 0; col < expected.shape().last(); col++) {
                assertEquals(expected.get(row, col), actual.get(row, col), tolerance,
                        label + " row=" + row + " col=" + col);
            }
        }
    }
}
