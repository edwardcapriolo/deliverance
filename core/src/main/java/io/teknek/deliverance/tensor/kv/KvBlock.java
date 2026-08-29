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
    private final AbstractTensor storage;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    KvBlock(int blockIndex, int blockSize, int tokenCount, int layers, int kvLength, AbstractTensor storage) {
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

    void copyKeyRow(int layer, int position, AbstractTensor destination) {
        copyRow(layer, position, 0, destination);
    }

    void copyValueRow(int layer, int position, AbstractTensor destination) {
        copyRow(layer, position, 1, destination);
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

    private AbstractTensor rowCopy(int layer, int position, int keyOrValue, TensorAllocator allocator) {
        requireOpen();
        AbstractTensor copy = allocator.getDirty(storage.dType(), TensorShape.of(1, kvLength));
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
        destination.copyFrom(storage, storage.getOffset(layer, keyOrValue, blockRow, 0), 0, kvLength);
    }

    private AbstractTensor rowView(int layer, int position, int keyOrValue) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(containsPosition(position), "position not in block");
        int blockRow = position - startPosition();
        return storage.slice(true, layer, keyOrValue, blockRow);
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
