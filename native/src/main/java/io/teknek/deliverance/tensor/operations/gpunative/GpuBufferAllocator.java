package io.teknek.deliverance.tensor.operations.gpunative;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;

import io.teknek.deliverance.tensor.operations.util.JarSupport;

/** Owns native WebGPU storage buffers without exposing native types to tensor2. */
public final class GpuBufferAllocator implements AutoCloseable, io.teknek.deliverance.tensor2.DeviceAllocator {
    private static boolean initialized;
    private final AtomicBoolean closed = new AtomicBoolean();

    public GpuBufferAllocator() {
        initializeGpu();
    }

    private static synchronized void initializeGpu() {
        if (initialized) {
            return;
        }
        if (!JarSupport.maybeLoadLibrary("webgpu_dawn")) {
            System.loadLibrary("webgpu_dawn");
        }
        if (!JarSupport.maybeLoadLibrary("deliverancegpu")) {
            System.loadLibrary("deliverancegpu");
        }
        LongBuffer results = ByteBuffer.allocateDirect(Long.BYTES * 3)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asLongBuffer();
        NativeGPU.init_gpu(MemorySegment.ofBuffer(results));
        if (results.get(0) < 0) {
            throw new IllegalStateException("WebGPU device initialization failed");
        }
        initialized = true;
    }

    /** Returns the device name used by this allocator. */
    @Override
    public String device() {
        return "gpu";
    }

    /** Allocates a zeroed or uninitialized WebGPU storage buffer. */
    public GpuBuffer allocateBuffer(int bytes, boolean zero) {
        if (closed.get()) {
            throw new IllegalStateException("GPU buffer allocator is closed");
        }
        if (bytes <= 0) {
            throw new IllegalArgumentException("bytes must be positive");
        }
        long handle = NativeGPU.allocate_gpu_buffer(bytes, zero);
        if (handle == -1L) {
            throw new IllegalStateException("WebGPU buffer allocation failed bytes=" + bytes);
        }
        return new GpuBuffer(handle, bytes, this);
    }

    /** Allocates GPU storage for a device-backed TensorRef. */
    @Override
    public long allocate(io.teknek.deliverance.DType dType, io.teknek.deliverance.tensor.TensorShape shape,
            boolean zero) {
        long bytes = shape.size() * dType.size();
        if (bytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("GPU tensor is too large: " + bytes);
        }
        return allocateBuffer((int) bytes, zero).handle();
    }

    @Override
    public void upload(long handle, MemorySegment source, long bytes) {
        if (closed.get()) {
            throw new IllegalStateException("GPU buffer allocator is closed");
        }
        if (bytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("GPU upload is too large: " + bytes);
        }
        NativeGPU.upload_gpu_buffer(handle, source, (int) bytes);
    }

    @Override
    public void download(long handle, MemorySegment destination, long bytes) {
        if (closed.get()) {
            throw new IllegalStateException("GPU buffer allocator is closed");
        }
        if (bytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("GPU download is too large: " + bytes);
        }
        if (NativeGPU.download_gpu_buffer(handle, destination, (int) bytes) != 0) {
            throw new IllegalStateException("GPU buffer download failed");
        }
    }

    /** Releases GPU storage for a device-backed TensorRef. */
    @Override
    public void release(long handle) {
        NativeGPU.free_gpu_buffer(handle);
    }

    private void release(GpuBuffer buffer) {
        if (buffer.owner != this) {
            throw new IllegalArgumentException("buffer belongs to another allocator");
        }
        NativeGPU.free_gpu_buffer(buffer.handle);
    }

    /** Releases all allocator-level resources; individual buffers remain explicitly owned. */
    @Override
    public void close() {
        closed.set(true);
    }

    /** Opaque WebGPU storage buffer handle. */
    public static final class GpuBuffer implements AutoCloseable {
        private final long handle;
        private final int bytes;
        private final GpuBufferAllocator owner;
        private final AtomicBoolean closed = new AtomicBoolean();

        private GpuBuffer(long handle, int bytes, GpuBufferAllocator owner) {
            this.handle = handle;
            this.bytes = bytes;
            this.owner = Objects.requireNonNull(owner, "owner");
        }

        /** Returns the opaque native buffer handle for provider submission. */
        public long handle() {
            return handle;
        }

        /** Returns the allocated byte capacity. */
        public int bytes() {
            return bytes;
        }

        /** Releases the underlying WebGPU buffer exactly once. */
        @Override
        public void close() {
            if (closed.compareAndSet(false, true)) {
                owner.release(this);
            }
        }
    }
}
