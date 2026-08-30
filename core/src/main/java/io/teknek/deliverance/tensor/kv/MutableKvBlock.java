package io.teknek.deliverance.tensor.kv;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
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
    private final KvBufferCacheSettings settings;
    private final TensorAllocator allocator;
    private final MetricRegistry metricRegistry;
    private boolean committed;

    MutableKvBlock(int blockIndex, int blockSize, int layers, int kvLength, DType dtype, TensorAllocator allocator,
            KvBufferCacheSettings settings, MetricRegistry metricRegistry) {
        this.blockIndex = blockIndex;
        this.blockSize = blockSize;
        this.layers = layers;
        this.kvLength = kvLength;
        this.allocator = allocator;
        this.settings = settings;
        this.metricRegistry = metricRegistry;
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

    AbstractTensor keyRowView(int layer, int position) {
        return rowView(layer, position, 0);
    }

    AbstractTensor valueRowView(int layer, int position) {
        return rowView(layer, position, 1);
    }

    void copyKeyRows(int layer, int positionStart, int rowCount, AbstractTensor destination, int destinationRowStart) {
        copyRows(layer, positionStart, rowCount, 0, destination, destinationRowStart);
    }

    void copyValueRows(int layer, int positionStart, int rowCount, AbstractTensor destination, int destinationRowStart) {
        copyRows(layer, positionStart, rowCount, 1, destination, destinationRowStart);
    }

    KvBlock commit(int tokenCount) {
        requireWritable();
        Preconditions.checkArgument(tokenCount >= 0 && tokenCount <= blockSize, "tokenCount out of bounds");
        committed = true;
        KvBlockStorage blockStorage = switch (settings.getKvBlockStoragePolicy()) {
            case DENSE -> new DenseKvBlockStorage(layers, tokenCount, blockSize, kvLength, storage);
            case MSE_TURBOQUANT -> tokenCount == blockSize
                    ? MseTurboQuantKvBlockStorage.encode(storage, layers, tokenCount, blockSize, kvLength,
                    settings.getKvTurboQuantBits(), allocator, metricRegistry)
                    : new DenseKvBlockStorage(layers, tokenCount, blockSize, kvLength, storage);
        };
        return new KvBlock(blockIndex, blockSize, tokenCount, layers, kvLength, blockStorage);
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

    private AbstractTensor rowView(int layer, int position, int keyOrValue) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(containsPosition(position), "position not in mutable block");
        int blockRow = position - startPosition();
        Preconditions.checkState(writtenRows.get(writtenIndex(layer, blockRow, keyOrValue)),
                "KV row has not been written");
        return storage.slice(true, layer, keyOrValue, blockRow);
    }

    private void copyRows(int layer, int positionStart, int rowCount, int keyOrValue, AbstractTensor destination,
            int destinationRowStart) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(rowCount >= 0, "rowCount must be >= 0");
        if (rowCount == 0) {
            return;
        }
        Preconditions.checkArgument(containsPosition(positionStart) && containsPosition(positionStart + rowCount - 1),
                "position range not in mutable block");
        Preconditions.checkArgument(destination.dims() == 2 && destination.shape().last() == kvLength,
                "destination must have kvLength columns");
        Preconditions.checkArgument(destinationRowStart >= 0
                && destinationRowStart + rowCount <= destination.shape().first(), "destination row range out of bounds");
        int blockRowStart = positionStart - startPosition();
        for (int i = 0; i < rowCount; i++) {
            Preconditions.checkState(writtenRows.get(writtenIndex(layer, blockRowStart + i, keyOrValue)),
                    "KV row has not been written");
        }
        destination.copyFrom(storage, storage.getOffset(layer, keyOrValue, blockRowStart, 0),
                destination.getOffset(destinationRowStart, 0), rowCount * kvLength);
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
