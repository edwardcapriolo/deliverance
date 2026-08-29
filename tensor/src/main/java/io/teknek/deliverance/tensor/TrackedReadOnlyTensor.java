package io.teknek.deliverance.tensor;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.util.zip.CRC32;

/**
 * Read-only tensor wrapper that detects hidden backing-storage mutation when closed.
 *
 * <p>This is intended for debug/assertion paths where callers borrow a tensor view that must not be mutated. Direct
 * tensor writes fail immediately, while writes through another alias or a {@link MemorySegment} are detected by the
 * checksum comparison.</p>
 */
public final class TrackedReadOnlyTensor extends AbstractTensor {
    private final AbstractTensor delegate;
    private final long initialChecksum;
    private boolean closed;

    public TrackedReadOnlyTensor(AbstractTensor delegate) {
        super(delegate.dType(), delegate.shape(), false);
        this.delegate = delegate;
        this.uid = delegate.getUid();
        delegate.locality().ifPresent(this::setLocality);
        this.initialChecksum = checksum(delegate);
    }

    public AbstractTensor delegate() {
        return delegate;
    }

    public boolean hasChecksumChanged() {
        return checksum(delegate) != initialChecksum;
    }

    @Override
    protected AbstractTensor make(TensorShape shape) {
        throw new UnsupportedOperationException("tracked read-only tensor cannot allocate derived storage");
    }

    @Override
    protected AbstractTensor make(int heapOffset, int heapLength, TensorShape shape, boolean cacheSlices) {
        return new TrackedReadOnlyTensor(delegate.make(heapOffset, heapLength, shape, cacheSlices));
    }

    @Override
    public float get(int... dims) {
        return delegate.get(dims);
    }

    @Override
    public float get(int row, int column) {
        return delegate.get(row, column);
    }

    @Override
    public void set(float v, int... dims) {
        throw new IllegalStateException("tracked read-only tensor cannot be written");
    }

    @Override
    public void set(float v, int row, int column) {
        throw new IllegalStateException("tracked read-only tensor cannot be written");
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
        throw new IllegalStateException("tracked read-only tensor cannot be written");
    }

    @Override
    public void clear() {
        throw new IllegalStateException("tracked read-only tensor cannot be written");
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        if (hasChecksumChanged()) {
            throw new IllegalStateException("tracked read-only tensor backing storage changed");
        }
    }

    private static long checksum(AbstractTensor tensor) {
        MemorySegment segment = tensor.getMemorySegment();
        int offset = tensor.getMemorySegmentOffset(0);
        long byteLength = segment.byteSize() - offset;
        if (byteLength < 0) {
            throw new IllegalStateException("tensor memory segment offset exceeds segment size");
        }
        ByteBuffer buffer = segment.asSlice(offset, byteLength).asByteBuffer();
        CRC32 crc32 = new CRC32();
        crc32.update(buffer.duplicate());
        return crc32.getValue();
    }
}
