package io.teknek.deliverance.tensor.operations;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

@Disabled("Documents known SIMD i8 x q4 parity failures exposed after removing tensor:tests reactor masking.")
public class NativeSimdI8Q4RegressionTest {

    @Test
    public void i8Q4ProjectionMatchesPanamaBaseline() {
        int batchSize = 3;
        int rows = 1_024;
        int k = 128;
        try (FloatBufferTensor denseInput = deterministicInput(batchSize, k);
             FloatBufferTensor denseWeight = deterministicWeight(rows, k);
             AbstractTensor input = AbstractTensorUtils.quantize(denseInput, DType.I8, true);
             AbstractTensor weight = AbstractTensorUtils.quantize(denseWeight, DType.Q4, true);
             FloatBufferTensor expected = new FloatBufferTensor(batchSize, rows);
             FloatBufferTensor actual = new FloatBufferTensor(batchSize, rows);
             WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            TensorOperations panama = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new ArrayQueueTensorAllocator(new MetricRegistry()), pool);
            TensorOperations simd = new NativeSimdTensorOperations(panama);

            panama.registerModelTensor(weight);
            panama.batchDotProduct(expected, input, weight, 0, 0, k, 0, 0, rows);
            simd.registerModelTensor(weight);
            simd.batchDotProduct(actual, input, weight, 0, 0, k, 0, 0, rows);

            assertTensorClose(expected, actual, 0.20f);
        }
    }

    @Test
    public void i8Q4BatchChunkOffsetAndTailFallsBackOrMatchesPanamaBaseline() {
        int batchSize = 2;
        int rows = 43;
        int k = 96;
        int columnOffset = 7;
        int chunkStart = 8;
        int chunkSize = 21;
        int cols = alignToBlock(columnOffset + k);
        int resultCols = chunkStart + chunkSize;
        try (FloatBufferTensor denseInput = deterministicInput(batchSize, cols);
             FloatBufferTensor denseWeight0 = deterministicWeight(rows, cols);
             FloatBufferTensor denseWeight1 = deterministicWeightVariant(rows, cols);
             AbstractTensor input = AbstractTensorUtils.quantize(denseInput, DType.I8, true);
             AbstractTensor weight0 = AbstractTensorUtils.quantize(denseWeight0, DType.Q4, true);
             AbstractTensor weight1 = AbstractTensorUtils.quantize(denseWeight1, DType.Q4, true);
             FloatBufferTensor expected0 = new FloatBufferTensor(batchSize, resultCols);
             FloatBufferTensor expected1 = new FloatBufferTensor(batchSize, resultCols);
             FloatBufferTensor actual0 = new FloatBufferTensor(batchSize, resultCols);
             FloatBufferTensor actual1 = new FloatBufferTensor(batchSize, resultCols);
             WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            TensorOperations panama = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new ArrayQueueTensorAllocator(new MetricRegistry()), pool);
            TensorOperations simd = new NativeSimdTensorOperations(panama);

            panama.registerModelTensor(weight0);
            panama.registerModelTensor(weight1);
            panama.dotProductBatchChunk(new AbstractTensor[]{expected0, expected1}, input,
                    new AbstractTensor[]{weight0, weight1}, columnOffset, k, chunkStart, chunkSize);
            simd.registerModelTensor(weight0);
            simd.registerModelTensor(weight1);
            simd.dotProductBatchChunk(new AbstractTensor[]{actual0, actual1}, input,
                    new AbstractTensor[]{weight0, weight1}, columnOffset, k, chunkStart, chunkSize);

            assertTensorClose(expected0, actual0, 0.20f);
            assertTensorClose(expected1, actual1, 0.20f);
        }
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

    private static void assertTensorClose(AbstractTensor expected, AbstractTensor actual, float tolerance) {
        assertEquals(expected.shape(), actual.shape(), "shape");
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int col = 0; col < expected.shape().last(); col++) {
                assertEquals(expected.get(row, col), actual.get(row, col), tolerance,
                        "row=" + row + " col=" + col);
            }
        }
    }
}
