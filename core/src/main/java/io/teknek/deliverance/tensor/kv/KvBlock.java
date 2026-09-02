package io.teknek.deliverance.tensor.kv;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;

import java.util.concurrent.atomic.AtomicBoolean;

/** Immutable committed KV block. */
public final class KvBlock implements AutoCloseable {
    private final int blockIndex;
    private final int blockSize;
    private final int tokenCount;
    private final int layers;
    private final int kvLength;
    private final KvBlockStorage storage;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    KvBlock(int blockIndex, int blockSize, int tokenCount, int layers, int kvLength, KvBlockStorage storage) {
        this.blockIndex = blockIndex;
        this.blockSize = blockSize;
        this.tokenCount = tokenCount;
        this.layers = layers;
        this.kvLength = kvLength;
        this.storage = storage;
        Preconditions.checkArgument(tokenCount >= 0 && tokenCount <= blockSize, "tokenCount out of bounds");
    }

    public int blockIndex() {
        return blockIndex;
    }

    public int blockSize() {
        return blockSize;
    }

    public int tokenCount() {
        return tokenCount;
    }

    public int startPosition() {
        return blockIndex * blockSize;
    }

    public int endPositionExclusive() {
        return startPosition() + tokenCount;
    }

    public boolean containsPosition(int position) {
        return position >= startPosition() && position < endPositionExclusive();
    }

    public KvBlockLayout layout() {
        return storage.layout();
    }

    public long denseBytesEquivalent() {
        return storage.denseBytesEquivalent();
    }

    public long encodedBytes() {
        return storage.encodedBytes();
    }

    boolean isClosed() {
        return closed.get();
    }

    void copyKeyRow(int layer, int position, AbstractTensor destination) {
        copyRow(layer, position, 0, destination);
    }

    void copyValueRow(int layer, int position, AbstractTensor destination) {
        copyRow(layer, position, 1, destination);
    }

    void copyKeyRows(int layer, int positionStart, int rowCount, AbstractTensor destination, int destinationRowStart) {
        copyRows(layer, positionStart, rowCount, 0, destination, destinationRowStart);
    }

    void copyValueRows(int layer, int positionStart, int rowCount, AbstractTensor destination, int destinationRowStart) {
        copyRows(layer, positionStart, rowCount, 1, destination, destinationRowStart);
    }

    AbstractTensor keyRowCopy(int layer, int position, TensorAllocator allocator) {
        return rowCopy(layer, position, 0, allocator);
    }

    AbstractTensor valueRowCopy(int layer, int position, TensorAllocator allocator) {
        return rowCopy(layer, position, 1, allocator);
    }

    AbstractTensor keyRowView(int layer, int position) {
        return rowView(layer, position, 0);
    }

    AbstractTensor valueRowView(int layer, int position) {
        return rowView(layer, position, 1);
    }

    AbstractTensor keyPageView(int layer) {
        requireOpen();
        validateLayer(layer);
        return storage.pageView(layer, 0);
    }

    AbstractTensor valuePageView(int layer) {
        requireOpen();
        validateLayer(layer);
        return storage.pageView(layer, 1);
    }

    private AbstractTensor rowCopy(int layer, int position, int keyOrValue, TensorAllocator allocator) {
        requireOpen();
        AbstractTensor copy = allocator.getDirty(storage.dtype(keyOrValue), TensorShape.of(1, kvLength));
        copyRow(layer, position, keyOrValue, copy);
        return copy;
    }

    private void copyRow(int layer, int position, int keyOrValue, AbstractTensor destination) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(containsPosition(position), "position not in block");
        Preconditions.checkArgument(destination.dims() == 2 && destination.shape().first() == 1
                && destination.shape().last() == kvLength, "destination must be [1, kvLength]");
        int blockRow = position - startPosition();
        storage.copyRow(layer, blockRow, keyOrValue, destination);
    }

    private void copyRows(int layer, int positionStart, int rowCount, int keyOrValue, AbstractTensor destination,
            int destinationRowStart) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(rowCount >= 0, "rowCount must be >= 0");
        if (rowCount == 0) {
            return;
        }
        Preconditions.checkArgument(positionStart >= startPosition()
                && positionStart + rowCount <= endPositionExclusive(), "position range not in block");
        storage.copyRows(layer, keyOrValue, positionStart - startPosition(), rowCount, destination, destinationRowStart);
    }

    private AbstractTensor rowView(int layer, int position, int keyOrValue) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(containsPosition(position), "position not in block");
        int blockRow = position - startPosition();
        return storage.rowView(layer, blockRow, keyOrValue);
    }

    private void validateLayer(int layer) {
        Preconditions.checkArgument(layer >= 0 && layer < layers, "layer out of bounds");
    }

    private void requireOpen() {
        if (closed.get()) {
            throw new IllegalStateException("KV block is closed");
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            storage.close();
        }
    }
}
