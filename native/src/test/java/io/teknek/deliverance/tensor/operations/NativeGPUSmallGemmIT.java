package io.teknek.deliverance.tensor.operations;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.gpunative.NativeGPU;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class NativeGPUSmallGemmIT {

    @Test
    void gpuTensorRegistryCanUnregisterAndReuseSlots() {
        NativeGPUTensorOperations gpu = loadGpuOperations();
        assumeTrue(gpu != null, "Native GPU operations are not available in this test environment");

        try (AbstractTensor first = tensor(1, 8, 1);
             AbstractTensor second = tensor(1, 8, 2);
             AbstractTensor third = tensor(1, 8, 3)) {
            long firstId = NativeGPU.register_tensor(first.getMemorySegment(), (int) first.getMemorySegment().byteSize());
            long secondId = NativeGPU.register_tensor(second.getMemorySegment(), (int) second.getMemorySegment().byteSize());
            assertTrue(firstId >= 0);
            assertTrue(secondId >= 0);
            NativeGPU.unregister_tensor(firstId);
            long reusedId = NativeGPU.register_tensor(third.getMemorySegment(), (int) third.getMemorySegment().byteSize());
            assertEquals(firstId, reusedId, "registry should reuse unregistered tensor slots");
            NativeGPU.unregister_tensor(secondId);
            NativeGPU.unregister_tensor(reusedId);
        }
    }

    @Test
    void gpuTensorRegistryFailsLoudlyInsteadOfOverflowing() {
        NativeGPUTensorOperations gpu = loadGpuOperations();
        assumeTrue(gpu != null, "Native GPU operations are not available in this test environment");

        List<Long> ids = new ArrayList<>();
        try (AbstractTensor tensor = tensor(1, 1, 1)) {
            long failed = Long.MIN_VALUE;
            for (int i = 0; i < 8_300; i++) {
                long id = NativeGPU.register_tensor(tensor.getMemorySegment(), (int) tensor.getMemorySegment().byteSize());
                if (id < 0) {
                    failed = id;
                    break;
                }
                ids.add(id);
            }
            assertEquals(-1L, failed, "registry should return -1 at capacity instead of overflowing");
        } finally {
            for (Long id : ids) {
                NativeGPU.unregister_tensor(id);
            }
        }
    }

    @Test
    void gpuSmallF32GemmReadbackMatchesCpuReference() {
        NativeGPUTensorOperations gpu = loadGpuOperations();
        assumeTrue(gpu != null, "Native GPU operations are not available in this test environment");

        try (AbstractTensor input = tensor(1, 8, 3);
             AbstractTensor weight = tensor(5, 8, 7);
             AbstractTensor actual = new FloatBufferTensor(1, 5);
             AbstractTensor expected = new FloatBufferTensor(1, 5)) {
            gpu.registerModelTensor(weight);
            new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new io.teknek.deliverance.tensor.ArrayQueueTensorAllocator(new io.dropwizard.metrics5.MetricRegistry()),
                    new io.teknek.deliverance.math.WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(1)))
                    .dotProductChunk(expected, input, weight, 0, 8, 0, 5);

            gpu.dotProductChunk(actual, input, weight, 0, 8, 0, 5);

            for (int col = 0; col < 5; col++) {
                assertEquals(expected.get(0, col), actual.get(0, col), 1.0e-4f, "col=" + col);
            }
        }
    }

    @Test
    void gpuF32GemmReadbackAtDecodeAttentionFailureSizes() {
        NativeGPUTensorOperations gpu = loadGpuOperations();
        assumeTrue(gpu != null, "Native GPU operations are not available in this test environment");

        assertGpuGemmMatchesReference(gpu, 132); // 528 bytes
        assertGpuGemmMatchesReference(gpu, 134); // 536 bytes: observed decode-attention failure size
        assertGpuGemmMatchesReference(gpu, 136); // 544 bytes
        assertGpuGemmMatchesReference(gpu, 172); // 688 bytes: observed benchmark failure size
    }

    @Test
    void gpuRepeatedF32GemmReadbackAtFailureSize() {
        NativeGPUTensorOperations gpu = loadGpuOperations();
        assumeTrue(gpu != null, "Native GPU operations are not available in this test environment");

        try (AbstractTensor input = tensor(1, 8, 3);
             AbstractTensor weight = tensor(134, 8, 7);
             AbstractTensor actual = new FloatBufferTensor(1, 134);
             AbstractTensor expected = new FloatBufferTensor(1, 134);
             io.teknek.deliverance.math.WrappedForkJoinPool pool =
                     new io.teknek.deliverance.math.WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(1))) {
            gpu.registerModelTensor(weight);
            new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new io.teknek.deliverance.tensor.ArrayQueueTensorAllocator(new io.dropwizard.metrics5.MetricRegistry()),
                    pool)
                    .dotProductChunk(expected, input, weight, 0, 8, 0, 134);

            for (int iteration = 0; iteration < 1_000; iteration++) {
                actual.clear();
                gpu.dotProductChunk(actual, input, weight, 0, 8, 0, 134);
                assertEquals(expected.get(0, 0), actual.get(0, 0), 1.0e-4f, "iteration=" + iteration);
                assertEquals(expected.get(0, 133), actual.get(0, 133), 1.0e-4f, "iteration=" + iteration);
            }
        }
    }

    @Test
    void gpuRepeatedPartialOffsetGemmIntoAttentionRowShape() {
        NativeGPUTensorOperations gpu = loadGpuOperations();
        assumeTrue(gpu != null, "Native GPU operations are not available in this test environment");

        try (AbstractTensor query = tensor(1, 8, 3);
             AbstractTensor page0 = tensor(32, 8, 7);
             AbstractTensor page1 = tensor(32, 8, 11);
             AbstractTensor page2 = tensor(32, 8, 13);
             AbstractTensor page3 = tensor(32, 8, 17);
             AbstractTensor page4 = tensor(6, 8, 19);
             AbstractTensor actual = new FloatBufferTensor(1, 134);
             AbstractTensor expected = new FloatBufferTensor(1, 134);
             io.teknek.deliverance.math.WrappedForkJoinPool pool =
                     new io.teknek.deliverance.math.WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(1))) {
            AbstractTensor[] pages = { page0, page1, page2, page3, page4 };
            TensorOperations reference = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new io.teknek.deliverance.tensor.ArrayQueueTensorAllocator(new io.dropwizard.metrics5.MetricRegistry()),
                    pool);
            for (AbstractTensor page : pages) {
                gpu.registerModelTensor(page);
            }
            writePages(reference, expected, query, pages);

            for (int iteration = 0; iteration < 500; iteration++) {
                actual.clear();
                writePages(gpu, actual, query, pages);
                assertEquals(expected.get(0, 0), actual.get(0, 0), 1.0e-4f, "iteration=" + iteration);
                assertEquals(expected.get(0, 33), actual.get(0, 33), 1.0e-4f, "iteration=" + iteration);
                assertEquals(expected.get(0, 133), actual.get(0, 133), 1.0e-4f, "iteration=" + iteration);
            }
        }
    }

    @Disabled("will take ~18 minutes to run without optimizations")
    void gpuDecodePagedAttentionRepeatedGrowingVisibleRows() {
        NativeGPUTensorOperations gpu = loadGpuOperations();
        assumeTrue(gpu != null, "Native GPU operations are not available in this test environment");

        int numberOfHeads = 32;
        int numberOfKeyValueHeads = 8;
        int headSize = 128;
        int kvLength = numberOfKeyValueHeads * headSize;
        int pageRows = 32;
        int maxVisibleRows = 160;
        int pageCount = (maxVisibleRows + pageRows - 1) / pageRows;
        try (AbstractTensor query = tensor(1, numberOfHeads * headSize, 3);
             AbstractTensor valueOut = new FloatBufferTensor(1, numberOfHeads * headSize)) {
            AbstractTensor[] keyPages = pages(pageCount, pageRows, kvLength, 7);
            AbstractTensor[] valuePages = pages(pageCount, pageRows, kvLength, 101);
            try {
                for (AbstractTensor page : keyPages) {
                    gpu.registerModelTensor(page);
                }
                for (int layer = 0; layer < 36; layer++) {
                    fill(query, layer + 3);
                    for (int visibleRows = 72; visibleRows <= maxVisibleRows; visibleRows++) {
                        if (visibleRows == 72 || visibleRows == 134 || visibleRows == maxVisibleRows
                                || visibleRows % 16 == 0) {
                            System.out.printf("GPU_DECODE_PAGED_ATTENTION_STRESS layer=%d visibleRows=%d%n",
                                    layer, visibleRows);
                        }
                        valueOut.clear();
                        gpu.decodePagedAttention(valueOut, query, keyPages, valuePages, visibleRows, numberOfHeads,
                                numberOfKeyValueHeads, headSize, 1.0f / (float) Math.sqrt(headSize), null);
                        assertFinite(valueOut.get(0, 0), "layer=" + layer + " visibleRows=" + visibleRows);
                        assertFinite(valueOut.get(0, numberOfHeads * headSize - 1),
                                "layer=" + layer + " visibleRows=" + visibleRows);
                    }
                }
            } finally {
                closeAll(keyPages);
                closeAll(valuePages);
            }
        }
    }

    private static void writePages(TensorOperations ops, AbstractTensor output, AbstractTensor query,
            AbstractTensor[] pages) {
        int offset = 0;
        for (AbstractTensor page : pages) {
            int rows = (int) page.shape().first();
            ops.batchDotProduct(output, query, page, 0, 0, 8, offset, 0, rows);
            offset += rows;
        }
    }

    private static void assertGpuGemmMatchesReference(NativeGPUTensorOperations gpu, int outputColumns) {
        try (AbstractTensor input = tensor(1, 8, 3);
             AbstractTensor weight = tensor(outputColumns, 8, 7);
             AbstractTensor actual = new FloatBufferTensor(1, outputColumns);
             AbstractTensor expected = new FloatBufferTensor(1, outputColumns);
             io.teknek.deliverance.math.WrappedForkJoinPool pool =
                     new io.teknek.deliverance.math.WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(1))) {
            gpu.registerModelTensor(weight);
            new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new io.teknek.deliverance.tensor.ArrayQueueTensorAllocator(new io.dropwizard.metrics5.MetricRegistry()),
                    pool)
                    .dotProductChunk(expected, input, weight, 0, 8, 0, outputColumns);

            gpu.dotProductChunk(actual, input, weight, 0, 8, 0, outputColumns);

            for (int col = 0; col < outputColumns; col++) {
                assertEquals(expected.get(0, col), actual.get(0, col), 1.0e-4f,
                        "outputColumns=" + outputColumns + " col=" + col);
            }
        }
    }

    private static NativeGPUTensorOperations loadGpuOperations() {
        try {
            return new NativeGPUTensorOperations();
        } catch (Throwable t) {
            return null;
        }
    }

    private static AbstractTensor tensor(int rows, int cols, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        fill(tensor, seed);
        return tensor;
    }

    private static AbstractTensor[] pages(int pageCount, int rows, int cols, int seed) {
        AbstractTensor[] pages = new AbstractTensor[pageCount];
        for (int i = 0; i < pageCount; i++) {
            pages[i] = tensor(rows, cols, seed + i * 17);
        }
        return pages;
    }

    private static void fill(AbstractTensor tensor, int seed) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int col = 0; col < tensor.shape().last(); col++) {
                tensor.set((((row * 19 + col * 23 + seed) % 43) - 21) / 21.0f, row, col);
            }
        }
    }

    private static void closeAll(AbstractTensor[] tensors) {
        for (AbstractTensor tensor : tensors) {
            tensor.close();
        }
    }

    private static void assertFinite(float value, String message) {
        if (!Float.isFinite(value)) {
            throw new AssertionError(message + " value=" + value);
        }
    }
}
