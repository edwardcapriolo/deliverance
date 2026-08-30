package io.teknek.deliverance.tensor;

import java.lang.foreign.MemorySegment;

/** Close-safe read-only wrapper for borrowed projection inputs. */
public final class ReadOnlyTensor extends AbstractTensor {
    private final AbstractTensor delegate;
    private final boolean closeDelegate;

    public ReadOnlyTensor(AbstractTensor delegate) {
        this(delegate, false);
    }

    public ReadOnlyTensor(AbstractTensor delegate, boolean closeDelegate) {
        super(delegate.dType(), delegate.shape(), false);
        this.delegate = delegate;
        this.closeDelegate = closeDelegate;
        this.uid = delegate.getUid();
        delegate.locality().ifPresent(this::setLocality);
    }

    public AbstractTensor delegate() {
        return delegate;
    }

    @Override
    protected AbstractTensor make(TensorShape shape) {
        throw new UnsupportedOperationException("read-only tensor cannot allocate derived storage");
    }

    @Override
    protected AbstractTensor make(int heapOffset, int heapLength, TensorShape shape, boolean cacheSlices) {
        return new ReadOnlyTensor(delegate.make(heapOffset, heapLength, shape, cacheSlices), true);
    }

    @Override
    public float get(int... dims) {
        return delegate.get(dims);
    }

    @Override
    public void set(float v, int... dims) {
        throw new UnsupportedOperationException("read-only tensor");
    }

    @Override
    public MemorySegment getMemorySegment() {
        return delegate.getMemorySegment();
    }

    @Override
    public int getMemorySegmentOffset(int offset) {
        return delegate.getMemorySegmentOffset(offset);
    }

    @Override
    public void copyFrom(AbstractTensor src, int srcOffset, int destOffset, int length) {
        throw new UnsupportedOperationException("read-only tensor");
    }

    @Override
    public void clear() {
        throw new UnsupportedOperationException("read-only tensor");
    }

    @Override
    public void close() {
        if (closeDelegate) {
            delegate.close();
        }
    }
}
