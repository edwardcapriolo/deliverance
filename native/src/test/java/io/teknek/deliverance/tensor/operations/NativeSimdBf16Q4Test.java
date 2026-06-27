package io.teknek.deliverance.tensor.operations;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.impl.BFloat16BufferTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q4ByteBufferTensor;
import com.codahale.metrics.MetricRegistry;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Optional;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class NativeSimdBf16Q4Test {

    @ParameterizedTest(name = "{0} m={1} rows={2} k={3} aOffset={4} bOffset={5}")
    @MethodSource("providerCases")
    public void bf16Q4MatchesNaiveReference(String providerName, TensorOperations ops,
            int batchSize, int rows, int k, int aOffset, int bOffset) {
        int inputCols = alignToBlock(k + aOffset);
        int weightCols = alignToBlock(k + bOffset);
        try (FloatBufferTensor denseInput = deterministicInput(batchSize, inputCols);
             FloatBufferTensor denseWeight = deterministicWeight(rows, weightCols);
             BFloat16BufferTensor bf16Input = new BFloat16BufferTensor(denseInput);
             Q4ByteBufferTensor q4Weight = (Q4ByteBufferTensor) AbstractTensorUtils.quantize(denseWeight, DType.Q4, true);
              FloatBufferTensor expected = new FloatBufferTensor(batchSize, rows);
              FloatBufferTensor actual = new FloatBufferTensor(batchSize, rows)) {
            new NaiveTensorOperations().batchDotProduct(expected, bf16Input, q4Weight, aOffset, bOffset, k, 0, 0, rows);
            ops.registerModelTensor(q4Weight);
            ops.batchDotProduct(actual, bf16Input, q4Weight, aOffset, bOffset, k, 0, 0, rows);
            assertTensorClose(expected, actual, 0.02f, providerName);
        }
    }

    @ParameterizedTest(name = "large bRowOffset={0}")
    @MethodSource("largeRowOffsets")
    public void bf16Q4HandlesLargeWeightRowOffsets(int bRowOffset) {
        int batchSize = 1;
        int rows = bRowOffset + 1;
        int k = 32;
        try (FloatBufferTensor denseInput = deterministicInput(batchSize, k);
             FloatBufferTensor denseWeight = deterministicWeight(rows, k);
             BFloat16BufferTensor bf16Input = new BFloat16BufferTensor(denseInput);
             Q4ByteBufferTensor q4Weight = (Q4ByteBufferTensor) AbstractTensorUtils.quantize(denseWeight, DType.Q4, true);
             FloatBufferTensor expected = new FloatBufferTensor(batchSize, rows);
             FloatBufferTensor actual = new FloatBufferTensor(batchSize, rows)) {
            new NaiveTensorOperations().batchDotProduct(expected, bf16Input, q4Weight, 0, 0, k, 0, bRowOffset, 1);
            new NativeSimdTensorOperations(new NaiveTensorOperations()).batchDotProduct(actual, bf16Input, q4Weight,
                    0, 0, k, 0, bRowOffset, 1);
            assertEquals(expected.get(0, bRowOffset), actual.get(0, bRowOffset), 0.02f);
        }
    }

    @ParameterizedTest(name = "batch {0} m={1} rows={2} k={3}")
    @MethodSource("providerBatchCases")
    public void bf16Q4BatchChunkMatchesSingleProjection(String providerName, TensorOperations ops,
            int batchSize, int rows, int k) {
        try (FloatBufferTensor denseInput = deterministicInput(batchSize, k);
             FloatBufferTensor denseWeight0 = deterministicWeight(rows, k);
             FloatBufferTensor denseWeight1 = deterministicWeight(rows, k);
             BFloat16BufferTensor bf16Input = new BFloat16BufferTensor(denseInput);
             Q4ByteBufferTensor q4Weight0 = (Q4ByteBufferTensor) AbstractTensorUtils.quantize(denseWeight0, DType.Q4, true);
             Q4ByteBufferTensor q4Weight1 = (Q4ByteBufferTensor) AbstractTensorUtils.quantize(denseWeight1, DType.Q4, true);
             FloatBufferTensor expected0 = new FloatBufferTensor(batchSize, rows);
              FloatBufferTensor expected1 = new FloatBufferTensor(batchSize, rows);
              FloatBufferTensor actual0 = new FloatBufferTensor(batchSize, rows);
              FloatBufferTensor actual1 = new FloatBufferTensor(batchSize, rows)) {
            ops.batchDotProduct(expected0, bf16Input, q4Weight0, 0, 0, k, 0, 0, rows);
            ops.batchDotProduct(expected1, bf16Input, q4Weight1, 0, 0, k, 0, 0, rows);
            ops.registerModelTensor(q4Weight0);
            ops.registerModelTensor(q4Weight1);
            ops.dotProductBatchChunk(new AbstractTensor[]{actual0, actual1}, bf16Input,
                    new AbstractTensor[]{q4Weight0, q4Weight1}, 0, k, 0, rows);
            assertTensorClose(expected0, actual0, 0.02f, providerName + " gate");
            assertTensorClose(expected1, actual1, 0.02f, providerName + " up");
        }
    }

    private static Stream<Arguments> providerCases() {
        return providers().flatMap(provider -> Stream.of(
                Arguments.of(1, 1, 32, 0, 0),
                Arguments.of(3, 4, 256, 0, 0),
                Arguments.of(2, 5, 128, 32, 32),
                Arguments.of(13, 17, 256, 0, 0)
        ).map(testCase -> prependProvider(provider, testCase)));
    }

    private static Stream<Arguments> largeRowOffsets() {
        return Stream.of(Arguments.of(70_000));
    }

    private static Stream<Arguments> providerBatchCases() {
        return providers().map(provider -> prependProvider(provider, Arguments.of(3, 4, 256)));
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
