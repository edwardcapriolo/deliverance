package io.teknek.deliverance.tensor.kv;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;

import java.util.BitSet;
import java.util.concurrent.atomic.AtomicBoolean;

final class MutableKvBlock implements AutoCloseable {
    private final int blockIndex;
    private final int blockSize;
    private final int layers;
    private final int kvLength;
    private final AbstractTensor storage;
    private final BitSet writtenRows;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private boolean committed;

    MutableKvBlock(int blockIndex, int blockSize, int layers, int kvLength, DType dtype, TensorAllocator allocator) {
        this.blockIndex = blockIndex;
        this.blockSize = blockSize;
        this.layers = layers;
        this.kvLength = kvLength;
        this.storage = allocator.getDirty(dtype, TensorShape.of(layers, 2, blockSize, kvLength));
        this.writtenRows = new BitSet(layers * blockSize * 2);
    }

    int blockIndex() {
        return blockIndex;
    }

    int startPosition() {
        return blockIndex * blockSize;
    }

    boolean containsPosition(int position) {
        return position >= startPosition() && position < startPosition() + blockSize;
    }

    void write(int layer, int position, AbstractTensor key, AbstractTensor value) {
        requireWritable();
        validateLayer(layer);
        Preconditions.checkArgument(containsPosition(position), "position not in mutable block");
        validateRow(key, "key");
        validateRow(value, "value");
        int blockRow = position - startPosition();
        storage.copyFrom(key, 0, storage.getOffset(layer, 0, blockRow, 0), kvLength);
        storage.copyFrom(value, 0, storage.getOffset(layer, 1, blockRow, 0), kvLength);
        writtenRows.set(writtenIndex(layer, blockRow, 0));
        writtenRows.set(writtenIndex(layer, blockRow, 1));
    }

    AbstractTensor keyRowCopy(int layer, int position, TensorAllocator allocator) {
        return rowCopy(layer, position, 0, allocator);
    }

    AbstractTensor valueRowCopy(int layer, int position, TensorAllocator allocator) {
        return rowCopy(layer, position, 1, allocator);
    }

    KvBlock commit(int tokenCount) {
        requireWritable();
        Preconditions.checkArgument(tokenCount >= 0 && tokenCount <= blockSize, "tokenCount out of bounds");
        committed = true;
        return new KvBlock(blockIndex, blockSize, tokenCount, layers, kvLength, storage);
    }

    private AbstractTensor rowCopy(int layer, int position, int keyOrValue, TensorAllocator allocator) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(containsPosition(position), "position not in mutable block");
        int blockRow = position - startPosition();
        Preconditions.checkState(writtenRows.get(writtenIndex(layer, blockRow, keyOrValue)),
                "KV row has not been written");
        AbstractTensor copy = allocator.getDirty(storage.dType(), TensorShape.of(1, kvLength));
        copy.copyFrom(storage, storage.getOffset(layer, keyOrValue, blockRow, 0), 0, kvLength);
        return copy;
    }

    private int writtenIndex(int layer, int blockRow, int keyOrValue) {
        return ((layer * blockSize) + blockRow) * 2 + keyOrValue;
    }

    private void validateLayer(int layer) {
        Preconditions.checkArgument(layer >= 0 && layer < layers, "layer out of bounds");
    }

    private void validateRow(AbstractTensor row, String name) {
        Preconditions.checkArgument(row.dims() == 2 && row.shape().first() == 1 && row.shape().last() == kvLength,
                name + " must be [1, kvLength]");
        Preconditions.checkArgument(row.dType() == storage.dType(), name + " dtype must match KV dtype");
    }

    private void requireWritable() {
        requireOpen();
        if (committed) {
            throw new IllegalStateException("KV block has been committed");
        }
    }

    private void requireOpen() {
        if (closed.get()) {
            throw new IllegalStateException("KV block is closed");
        }
    }

    @Override
    public void close() {
        if (!committed && closed.compareAndSet(false, true)) {
            storage.close();
        }
    }
}
