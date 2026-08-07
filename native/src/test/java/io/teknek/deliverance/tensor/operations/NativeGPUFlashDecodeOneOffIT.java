package io.teknek.deliverance.tensor.operations;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

//@Disabled("Manual one-off Qwen-sized GPU flash decode smoke/benchmark.")
class NativeGPUFlashDecodeOneOffIT {

    @Test
    void qwenShapeOneDecodeMatchesCpuAndPrintsTiming() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(16))) {
            TensorOperations cpu = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new ArrayQueueTensorAllocator(new MetricRegistry()), pool);
            NativeGPUTensorOperations gpu = loadGpuOperations();
            assumeTrue(gpu != null, "Native GPU operations are not available in this test environment");

            int numberOfHeads = 32;
            int numberOfKeyValueHeads = 8;
            int headSize = 128;
            int kvLength = numberOfKeyValueHeads * headSize;
            int pageRows = 32;
            int visibleRows = Integer.getInteger("deliverance.gpu.flash.visibleRows", 256);
            int pageCount = (visibleRows + pageRows - 1) / pageRows;
            float scale = 1.0f / (float) Math.sqrt(headSize);

            try (AbstractTensor query = tensor(1, numberOfHeads * headSize, 31);
                 AbstractTensor expected = new FloatBufferTensor(1, numberOfHeads * headSize);
                 AbstractTensor actual = new FloatBufferTensor(1, numberOfHeads * headSize)) {
                AbstractTensor[] keyPages = pages(pageCount, pageRows, kvLength, 43);
                AbstractTensor[] valuePages = pages(pageCount, pageRows, kvLength, 97);
                try {
                    assumeTrue(gpu.supportsFlashDecodePagedAttention(actual, query, keyPages, valuePages,
                            visibleRows, numberOfHeads, numberOfKeyValueHeads, headSize, scale, null),
                            "GPU flash decode does not support this shape");

                    long cpuNanos = time(() -> CausalFlashReference.flashDecode(cpu, expected, query, keyPages,
                            valuePages, visibleRows, numberOfHeads, numberOfKeyValueHeads, headSize, scale));
                    long gpuNanos = time(() -> gpu.flashDecodePagedAttention(actual, query, keyPages, valuePages,
                            visibleRows, numberOfHeads, numberOfKeyValueHeads, headSize, scale, null));

                    assertAllValuesClose(expected, actual, 1.0e-3f);
                    System.out.printf(java.util.Locale.ROOT,
                            "qwen_gpu_flash_one_decode visibleRows=%d pages=%d cpu_us=%.3f gpu_us=%.3f gpu_vs_cpu=%.4f%n",
                            visibleRows, pageCount, cpuNanos / 1_000.0, gpuNanos / 1_000.0,
                            (double) cpuNanos / (double) gpuNanos);
                } finally {
                    closeAll(keyPages);
                    closeAll(valuePages);
                }
            }
        }
    }

    private static long time(Runnable runnable) {
        long start = System.nanoTime();
        runnable.run();
        return System.nanoTime() - start;
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

    private static void assertAllValuesClose(AbstractTensor expected, AbstractTensor actual, float tolerance) {
        assertEquals(expected.shape(), actual.shape());
        for (int col = 0; col < expected.shape().last(); col++) {
            assertEquals(expected.get(0, col), actual.get(0, col), tolerance, "col=" + col);
        }
    }

    private static void closeAll(AbstractTensor[] tensors) {
        for (AbstractTensor tensor : tensors) {
            tensor.close();
        }
    }

    private static final class CausalFlashReference {
        private static void flashDecode(TensorOperations cpu, AbstractTensor out, AbstractTensor query,
                AbstractTensor[] keyPages, AbstractTensor[] valuePages, int visibleRows, int numberOfHeads,
                int numberOfKeyValueHeads, int headSize, float scale) {
            cpu.decodePagedAttention(out, query, keyPages, valuePages, visibleRows, numberOfHeads,
                    numberOfKeyValueHeads, headSize, scale, null);
        }
    }
}
