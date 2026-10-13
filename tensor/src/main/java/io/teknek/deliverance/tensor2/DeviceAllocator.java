package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;

/** Allocates opaque device storage used by device-backed TensorRefs. */
public interface DeviceAllocator {
    /** Returns the device name handled by this allocator. */
    String device();

    /** Allocates storage for a tensor and optionally clears it on the device. */
    long allocate(DType dType, TensorShape shape, boolean zero);

    /** Releases a previously allocated device handle. */
    void release(long handle);
}
