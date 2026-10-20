package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.tensor.operations.gpunative.GpuBufferAllocator;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Integration coverage for native WebGPU storage-buffer allocation and lifetime handling. */
class GpuBufferAllocatorTest {
    /** Allocates a zero-initialized GPU buffer and releases it explicitly. */
    @Test
    void allocatesZeroedBufferWithExpectedHandleAndSize() {
        GpuBufferAllocator allocator = availableAllocator();
        try (allocator; GpuBufferAllocator.GpuBuffer buffer = allocator.allocateBuffer(4096, true)) {
            assertTrue(buffer.handle() != 0L);
            assertEquals(4096, buffer.bytes());
        }
    }

    /** Rejects invalid allocation sizes and use after allocator shutdown. */
    @Test
    void validatesAllocationLifecycle() {
        GpuBufferAllocator allocator = availableAllocator();
        assertThrows(IllegalArgumentException.class, () -> allocator.allocateBuffer(0, false));
        allocator.close();
        assertThrows(IllegalStateException.class, () -> allocator.allocateBuffer(16, false));
    }

    /** Verifies that Lighter routes a GPU allocation through the injected device allocator. */
    @Test
    void lighterAllocatesGpuBackedTensorRef() {
        GpuBufferAllocator allocator = availableAllocator();
        try (allocator) {
            Lighter lighter = new Lighter(new MetricRegistry());
            lighter.registerDeviceAllocator(allocator);
            try (TensorRef tensor = lighter.allocate(DType.F32, TensorShape.of(1, 16), "gpu")) {
                assertEquals("gpu", tensor.device());
                assertTrue(tensor.deviceHandle() != 0L);
                assertEquals(null, tensor.memorySegment());
            }
        }
    }

    /** Verifies that Lighter copies tensor storage to GPU and back to CPU. */
    @Test
    void lighterCopiesCpuTensorToGpu() {
        GpuBufferAllocator allocator = availableAllocator();
        try (allocator) {
            Lighter lighter = new Lighter(new MetricRegistry());
            lighter.registerDeviceAllocator(allocator);
            try (TensorRef source = lighter.allocate(DType.F32, TensorShape.of(1, 4));
                 TensorRef target = fillAndCopyToGpu(lighter, source);
                 TensorRef result = lighter.to(target, "cpu")) {
                assertEquals("gpu", target.device());
                assertTrue(target.deviceHandle() != 0L);
                assertEquals("cpu", result.device());
                for (int column = 0; column < 4; column++) {
                    assertEquals(column + 1.0f, result.get(0, column));
                }
            }
        }
    }

    private static TensorRef fillAndCopyToGpu(Lighter lighter, TensorRef source) {
        source.set(1.0f, 0, 0);
        source.set(2.0f, 0, 1);
        source.set(3.0f, 0, 2);
        source.set(4.0f, 0, 3);
        return lighter.to(source, "gpu");
    }

    /** Creates an allocator when WebGPU is available and skips this IT otherwise. */
    private static GpuBufferAllocator availableAllocator() {
        try {
            GpuBufferAllocator allocator = new GpuBufferAllocator();
            try (GpuBufferAllocator.GpuBuffer ignored = allocator.allocateBuffer(8, false)) {
                return allocator;
            }
        } catch (Throwable failure) {
            Assumptions.assumeTrue(false, "WebGPU unavailable: " + failure);
            throw new AssertionError(failure);
        }
    }
}
