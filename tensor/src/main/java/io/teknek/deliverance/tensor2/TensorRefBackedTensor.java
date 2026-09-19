package io.teknek.deliverance.tensor2;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;

import java.lang.foreign.MemorySegment;

/**
 * Temporary bridge that lets old tensor code consume a tensor2-owned result.
 */
public final class TensorRefBackedTensor extends AbstractTensor {
    private final TensorRef ref;

    public TensorRefBackedTensor(TensorRef ref) {
        super(ref.dType(), ref.shape(), false);
        this.ref = java.util.Objects.requireNonNull(ref, "ref");
    }

    @Override
    protected AbstractTensor make(TensorShape shape) {
        throw new UnsupportedOperationException("TensorRef-backed tensor does not support creating views");
    }

    @Override
    protected AbstractTensor make(int heapOffset, int heapLength, TensorShape shape, boolean cacheSlices) {
        throw new UnsupportedOperationException("TensorRef-backed tensor does not support creating views");
    }

    @Override
    public float get(int... dims) {
        return ref.underlying().get(dims);
    }

    @Override
    public float get(int row, int column) {
        return ref.underlying().get(row, column);
    }

    @Override
    public void set(float v, int... dims) {
        ref.underlying().set(v, dims);
    }

    @Override
    public void set(float v, int row, int column) {
        ref.underlying().set(v, row, column);
    }

    @Override
    public MemorySegment getMemorySegment() {
        return ref.underlying().getMemorySegment();
    }

    @Override
    public int getMemorySegmentOffset(int offset) {
        return ref.underlying().getMemorySegmentOffset(offset);
    }

    @Override
    public void copyFrom(AbstractTensor src, int srcOffset, int destOffset, int length) {
        Preconditions.checkArgument(src.dType() == dType(), "different types");
        long bytes = (long) length * dType().size();
        getMemorySegment().asSlice(getMemorySegmentOffset(destOffset), bytes)
                .copyFrom(src.getMemorySegment().asSlice(src.getMemorySegmentOffset(srcOffset), bytes));
    }

    @Override
    public void clear() {
        getMemorySegment().fill((byte) 0);
    }

    @Override
    public void close() {
        ref.close();
    }
}
