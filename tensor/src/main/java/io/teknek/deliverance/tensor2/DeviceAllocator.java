package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;

import java.lang.foreign.MemorySegment;

/** Allocates opaque device storage used by device-backed TensorRefs. */
public interface DeviceAllocator {
    /** Returns the device name handled by this allocator. */
    String device();

    /** Allocates storage for a tensor and optionally clears it on the device. */
    long allocate(DType dType, TensorShape shape, boolean zero);

    /** Copies host bytes into an allocated device buffer. */
    default void upload(long handle, MemorySegment source, long bytes) {
        throw new UnsupportedOperationException("Host-to-device upload is not supported for " + device());
    }

    /** Copies device bytes into host storage. */
    default void download(long handle, MemorySegment destination, long bytes) {
        throw new UnsupportedOperationException("Device-to-host download is not supported for " + device());
    }

    /** Releases a previously allocated device handle. */
    void release(long handle);
}
