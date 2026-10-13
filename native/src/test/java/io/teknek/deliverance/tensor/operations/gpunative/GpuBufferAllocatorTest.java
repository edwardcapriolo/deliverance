package io.teknek.deliverance.tensor.operations.gpunative;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

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

    /** Creates an allocator when WebGPU is available and skips this IT otherwise. */
    private static GpuBufferAllocator availableAllocator() {
        try {
            long probe = NativeGPU.allocate_gpu_buffer(8, false);
            NativeGPU.free_gpu_buffer(probe);
            return new GpuBufferAllocator();
        } catch (Throwable failure) {
            Assumptions.assumeTrue(false, "WebGPU unavailable: " + failure);
            throw new AssertionError(failure);
        }
    }
}
