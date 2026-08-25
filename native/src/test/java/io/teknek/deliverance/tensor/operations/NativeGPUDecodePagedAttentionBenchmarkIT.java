package io.teknek.deliverance.tensor.operations;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ForkJoinPool;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

@Disabled("Manual GPU decode attention microbenchmark; prints timings for planning GPU KV work.")
class NativeGPUDecodePagedAttentionBenchmarkIT {

    @Test
    void compareNativeSimdAndGpuPackedDecodeAttention() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(16))) {
            TensorOperations cpu = new NativeSimdTensorOperations(new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new ArrayQueueTensorAllocator(new MetricRegistry()), pool));
            TensorOperations gpu = loadGpuOperations();
            assumeTrue(gpu != null, "Native GPU operations are not available in this test environment");

            int numberOfHeads = 32;
            int numberOfKeyValueHeads = 8;
            int headSize = 128;
            int kvLength = numberOfKeyValueHeads * headSize;
            int pageRows = 32;
            int warmup = 5;
            int iterations = 20;

            System.out.println("visibleRows,cpu_us,gpu_packed_us,gpu_vs_cpu");
            for (int visibleRows : new int[] { 128, 256, 512, 1024 }) {
                int pageCount = (visibleRows + pageRows - 1) / pageRows;
                try (AbstractTensor query = tensor(1, numberOfHeads * headSize, 31);
                     AbstractTensor cpuOut = new FloatBufferTensor(1, numberOfHeads * headSize);
                     AbstractTensor gpuOut = new FloatBufferTensor(1, numberOfHeads * headSize)) {
                    AbstractTensor[] keyPages = pages(pageCount, pageRows, kvLength, 43);
                    AbstractTensor[] valuePages = pages(pageCount, pageRows, kvLength, 97);
                    try {
                        float scale = 1.0f / (float) Math.sqrt(headSize);
                        for (int i = 0; i < warmup; i++) {
                            cpu.decodePagedAttention(cpuOut, query, keyPages, valuePages, visibleRows, numberOfHeads,
                                    numberOfKeyValueHeads, headSize, scale, null);
                            gpu.decodePagedAttention(gpuOut, query, keyPages, valuePages, visibleRows, numberOfHeads,
                                    numberOfKeyValueHeads, headSize, scale, null);
                        }

                        long cpuNanos = time(iterations, () -> {
                            cpuOut.clear();
                            cpu.decodePagedAttention(cpuOut, query, keyPages, valuePages, visibleRows, numberOfHeads,
                                    numberOfKeyValueHeads, headSize, scale, null);
                        });
                        long gpuNanos = time(iterations, () -> {
                            gpuOut.clear();
                            gpu.decodePagedAttention(gpuOut, query, keyPages, valuePages, visibleRows, numberOfHeads,
                                    numberOfKeyValueHeads, headSize, scale, null);
                        });
                        double cpuUs = cpuNanos / 1_000.0 / iterations;
                        double gpuUs = gpuNanos / 1_000.0 / iterations;
                        System.out.printf(java.util.Locale.ROOT, "%d,%.3f,%.3f,%.4f%n", visibleRows, cpuUs, gpuUs,
                                cpuUs / gpuUs);
                    } finally {
                        closeAll(keyPages);
                        closeAll(valuePages);
                    }
                }
            }
        }
    }

    private static long time(int iterations, Runnable runnable) {
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            runnable.run();
        }
        return System.nanoTime() - start;
    }

    private static TensorOperations loadGpuOperations() {
        try {
            return new NativeGPUTensorOperations();
        } catch (Throwable t) {
            return null;
        }
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

    private static AbstractTensor[] pages(int pageCount, int rows, int cols, int seed) {
        AbstractTensor[] pages = new AbstractTensor[pageCount];
        for (int i = 0; i < pageCount; i++) {
            pages[i] = tensor(rows, cols, seed + i * 17);
        }
        return pages;
    }

    private static void closeAll(AbstractTensor[] tensors) {
        for (AbstractTensor tensor : tensors) {
            tensor.close();
        }
    }
}
