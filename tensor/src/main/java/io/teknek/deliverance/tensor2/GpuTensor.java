package io.teknek.deliverance.tensor2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;

import java.lang.foreign.MemorySegment;

/** Tensor storage backed by an opaque GPU allocation rather than CPU memory. */
final class GpuTensor extends Tensor {
    private final long handle;

    GpuTensor(long handle, DType dType, TensorShape shape, int stride) {
        this.handle = handle;
    }

    @Override
    long deviceHandle() {
        return handle;
    }

    @Override
    public float get(int... dims) {
        throw new UnsupportedOperationException("GPU tensor scalar reads require explicit readback");
    }

    @Override
    public float get(int row, int column) {
        throw new UnsupportedOperationException("GPU tensor scalar reads require explicit readback");
    }

    @Override
    public void set(float value, int row, int column) {
        throw new UnsupportedOperationException("GPU tensor scalar writes require a device operation");
    }

    @Override
    public void set(float value, int... dims) {
        throw new UnsupportedOperationException("GPU tensor scalar writes require a device operation");
    }

    @Override
    public MemorySegment getMemorySegment() {
        return null;
    }

    @Override
    public int getMemorySegmentOffset(int offset) {
        return 0;
    }
}
