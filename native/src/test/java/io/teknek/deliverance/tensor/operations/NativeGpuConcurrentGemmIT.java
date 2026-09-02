package io.teknek.deliverance.tensor.operations;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class NativeGpuConcurrentGemmIT {

    @Test
    void concurrentQkvLikeGpuGemmSubmissionsUseShadersAndRemainStable() throws Exception {
        NativeGPUTensorOperations gpu = loadGpuOperations();
        assumeTrue(gpu != null, "Native GPU operations are not available in this test environment");

        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(3));
             AbstractTensor input = deterministicInput(384, 256);
             AbstractTensor qWeight = q4Weight(256, 256, 11);
             AbstractTensor kWeight = q4Weight(128, 256, 17);
             AbstractTensor vWeight = q4Weight(128, 256, 23);
             AbstractTensor qActual = new FloatBufferTensor(384, 256);
             AbstractTensor kActual = new FloatBufferTensor(384, 128);
             AbstractTensor vActual = new FloatBufferTensor(384, 128);
             AbstractTensor qExpected = new FloatBufferTensor(384, 256);
             AbstractTensor kExpected = new FloatBufferTensor(384, 128);
             AbstractTensor vExpected = new FloatBufferTensor(384, 128)) {
            TensorOperations panama = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    new ArrayQueueTensorAllocator(new MetricRegistry()), pool);
            gpu.registerModelTensor(qWeight);
            gpu.registerModelTensor(kWeight);
            gpu.registerModelTensor(vWeight);

            panama.batchDotProduct(qExpected, input, qWeight, 0, 0, 256, 0, 0, 256);
            panama.batchDotProduct(kExpected, input, kWeight, 0, 0, 256, 0, 0, 128);
            panama.batchDotProduct(vExpected, input, vWeight, 0, 0, 256, 0, 0, 128);

            long gpuCallsBefore = gpu.gemmGpuCalls();
            long fallbackCallsBefore = gpu.gemmFallbackCalls();
            for (int iteration = 0; iteration < 20; iteration++) {
                CountDownLatch ready = new CountDownLatch(3);
                CountDownLatch start = new CountDownLatch(1);
                AtomicReference<Throwable> failure = new AtomicReference<>();
                Future<?> q = submit(pool, ready, start, failure,
                        () -> gpu.batchDotProduct(qActual, input, qWeight, 0, 0, 256, 0, 0, 256));
                Future<?> k = submit(pool, ready, start, failure,
                        () -> gpu.batchDotProduct(kActual, input, kWeight, 0, 0, 256, 0, 0, 128));
                Future<?> v = submit(pool, ready, start, failure,
                        () -> gpu.batchDotProduct(vActual, input, vWeight, 0, 0, 256, 0, 0, 128));
                assertTrue(ready.await(10, TimeUnit.SECONDS), "workers did not become ready");
                start.countDown();
                q.get(30, TimeUnit.SECONDS);
                k.get(30, TimeUnit.SECONDS);
                v.get(30, TimeUnit.SECONDS);
                assertNull(failure.get(), "worker failed");

                assertSampleClose(qExpected, qActual, 0.08f, "q iteration=" + iteration);
                assertSampleClose(kExpected, kActual, 0.08f, "k iteration=" + iteration);
                assertSampleClose(vExpected, vActual, 0.08f, "v iteration=" + iteration);
            }

            assertEquals(60, gpu.gemmGpuCalls() - gpuCallsBefore, "expected all submissions to use GPU GEMM shaders");
            assertEquals(0, gpu.gemmFallbackCalls() - fallbackCallsBefore, "expected no GPU fallback");
        } finally {
            gpu.close();
        }
    }

    @Test
    void concurrentQwen4bPrefillQkvGpuGemmSubmissionsUseShadersAndRemainStable() throws Exception {
        NativeGPUTensorOperations gpu = loadGpuOperations();
        assumeTrue(gpu != null, "Native GPU operations are not available in this test environment");

        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(3));
             AbstractTensor input = deterministicInput(512, 2560);
             AbstractTensor qWeight = q4Weight(2560, 2560, 31);
             AbstractTensor kWeight = q4Weight(1024, 2560, 37);
             AbstractTensor vWeight = q4Weight(1024, 2560, 41);
             AbstractTensor qActual = new FloatBufferTensor(512, 2560);
             AbstractTensor kActual = new FloatBufferTensor(512, 1024);
             AbstractTensor vActual = new FloatBufferTensor(512, 1024)) {
            gpu.registerModelTensor(qWeight);
            gpu.registerModelTensor(kWeight);
            gpu.registerModelTensor(vWeight);

            long gpuCallsBefore = gpu.gemmGpuCalls();
            long fallbackCallsBefore = gpu.gemmFallbackCalls();
            for (int iteration = 0; iteration < 5; iteration++) {
                qActual.clear();
                kActual.clear();
                vActual.clear();
                CountDownLatch ready = new CountDownLatch(3);
                CountDownLatch start = new CountDownLatch(1);
                AtomicReference<Throwable> failure = new AtomicReference<>();
                Future<?> q = submit(pool, ready, start, failure,
                        () -> gpu.batchDotProduct(qActual, input, qWeight, 0, 0, 2560, 0, 0, 2560));
                Future<?> k = submit(pool, ready, start, failure,
                        () -> gpu.batchDotProduct(kActual, input, kWeight, 0, 0, 2560, 0, 0, 1024));
                Future<?> v = submit(pool, ready, start, failure,
                        () -> gpu.batchDotProduct(vActual, input, vWeight, 0, 0, 2560, 0, 0, 1024));
                assertTrue(ready.await(10, TimeUnit.SECONDS), "workers did not become ready");
                start.countDown();
                q.get(120, TimeUnit.SECONDS);
                k.get(120, TimeUnit.SECONDS);
                v.get(120, TimeUnit.SECONDS);
                assertNull(failure.get(), "worker failed");

                assertFinite(qActual.get(0, 0), "q first iteration=" + iteration);
                assertFinite(qActual.get(511, 2559), "q last iteration=" + iteration);
                assertFinite(kActual.get(0, 0), "k first iteration=" + iteration);
                assertFinite(kActual.get(511, 1023), "k last iteration=" + iteration);
                assertFinite(vActual.get(0, 0), "v first iteration=" + iteration);
                assertFinite(vActual.get(511, 1023), "v last iteration=" + iteration);
            }

            assertEquals(15, gpu.gemmGpuCalls() - gpuCallsBefore, "expected all Qwen-shaped submissions to use GPU GEMM shaders");
            assertEquals(0, gpu.gemmFallbackCalls() - fallbackCallsBefore, "expected no GPU fallback");
        } finally {
            gpu.close();
        }
    }

    private static Future<?> submit(WrappedForkJoinPool pool, CountDownLatch ready, CountDownLatch start,
            AtomicReference<Throwable> failure, Runnable action) {
        return pool.getUnderlying().submit(() -> {
            try {
                ready.countDown();
                start.await();
                action.run();
            } catch (Throwable t) {
                failure.compareAndSet(null, t);
                throw new RuntimeException(t);
            }
        });
    }

    private static NativeGPUTensorOperations loadGpuOperations() {
        try {
            return new NativeGPUTensorOperations();
        } catch (Throwable t) {
            return null;
        }
    }

    private static AbstractTensor deterministicInput(int rows, int cols) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set(((row * 17 + col * 31) % 257 - 128) / 64.0f, row, col);
            }
        }
        return tensor;
    }

    private static AbstractTensor q4Weight(int rows, int cols, int seed) {
        FloatBufferTensor dense = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                dense.set(((row * 43 + col * 19 + seed) % 251 - 125) / 80.0f, row, col);
            }
        }
        try (dense) {
            return AbstractTensorUtils.quantize(dense, DType.Q4, true);
        }
    }

    private static void assertSampleClose(AbstractTensor expected, AbstractTensor actual, float tolerance, String label) {
        int lastRow = (int) expected.shape().first() - 1;
        int lastCol = (int) expected.shape().last() - 1;
        int midRow = lastRow / 2;
        int midCol = lastCol / 2;
        assertEquals(expected.get(0, 0), actual.get(0, 0), tolerance, label + " [0,0]");
        assertEquals(expected.get(midRow, midCol), actual.get(midRow, midCol), tolerance, label + " mid");
        assertEquals(expected.get(lastRow, lastCol), actual.get(lastRow, lastCol), tolerance, label + " last");
    }

    private static void assertFinite(float value, String label) {
        assertTrue(Float.isFinite(value), label + " should be finite");
    }
}
