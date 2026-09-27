package io.teknek.deliverance.tensor.kv;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

final class DenseKvBlockStorage implements KvBlockStorage {
    private final int layers;
    private final int tokenCount;
    private final int blockSize;
    private final int kvLength;
    private final TensorRef keyStorage;
    private final TensorRef valueStorage;
    private final Lighter lighter;

    DenseKvBlockStorage(int layers, int tokenCount, int blockSize, int kvLength, TensorRef keyStorage,
            TensorRef valueStorage, Lighter lighter) {
        this.layers = layers;
        this.tokenCount = tokenCount;
        this.blockSize = blockSize;
        this.kvLength = kvLength;
        this.keyStorage = keyStorage;
        this.valueStorage = valueStorage;
        this.lighter = lighter;
    }

    @Override
    public KvBlockLayout layout() {
        return KvBlockLayout.DENSE;
    }

    @Override
    public DType dtype() {
        return keyStorage.dType();
    }

    @Override
    public DType dtype(int keyOrValue) {
        return storage(keyOrValue).dType();
    }

    @Override
    public int layers() {
        return layers;
    }

    @Override
    public int tokenCount() {
        return tokenCount;
    }

    @Override
    public int blockSize() {
        return blockSize;
    }

    @Override
    public int kvLength() {
        return kvLength;
    }

    @Override
    public long denseBytesEquivalent() {
        return (long) layers * tokenCount * kvLength * keyStorage.dType().size()
                + (long) layers * tokenCount * kvLength * valueStorage.dType().size();
    }

    @Override
    public long encodedBytes() {
        return denseBytesEquivalent();
    }

    @Override
    public TensorRef rowView(int layer, int blockRow, int keyOrValue) {
        validate(layer, blockRow, keyOrValue);
        return storage(keyOrValue).slice(layer, blockRow);
    }

    @Override
    public TensorRef pageView(int layer, int keyOrValue) {
        Preconditions.checkArgument(layer >= 0 && layer < layers, "layer out of bounds");
        Preconditions.checkArgument(keyOrValue == 0 || keyOrValue == 1, "keyOrValue must be 0 or 1");
        return storage(keyOrValue).slice(layer);
    }

    @Override
    public void copyRow(int layer, int blockRow, int keyOrValue, TensorRef destination) {
        validate(layer, blockRow, keyOrValue);
        TensorRef storage = storage(keyOrValue);
        Preconditions.checkArgument(destination.dType() == storage.dType(), "destination dtype must match KV dtype");
        lighter.copy(storage, storage.shape().getOffset(layer, blockRow, 0), destination, 0, kvLength);
    }

    @Override
    public void copyRows(int layer, int keyOrValue, int blockRowStart, int rowCount, TensorRef destination,
            int destinationRowStart) {
        validateRange(layer, keyOrValue, blockRowStart, rowCount, destination, destinationRowStart);
        if (rowCount == 0) {
            return;
        }
        TensorRef storage = storage(keyOrValue);
        Preconditions.checkArgument(destination.dType() == storage.dType(), "destination dtype must match KV dtype");
        lighter.copy(storage, storage.shape().getOffset(layer, blockRowStart, 0), destination,
                destination.shape().getOffset(destinationRowStart, 0), rowCount * kvLength);
    }

    private TensorRef storage(int keyOrValue) {
        return keyOrValue == 0 ? keyStorage : valueStorage;
    }

    TensorRef keyStorage() {
        return keyStorage;
    }

    TensorRef valueStorage() {
        return valueStorage;
    }

    private void validate(int layer, int blockRow, int keyOrValue) {
        Preconditions.checkArgument(layer >= 0 && layer < layers, "layer out of bounds");
        Preconditions.checkArgument(blockRow >= 0 && blockRow < tokenCount, "blockRow out of bounds");
        Preconditions.checkArgument(keyOrValue == 0 || keyOrValue == 1, "keyOrValue must be 0 or 1");
    }

    private void validateRange(int layer, int keyOrValue, int blockRowStart, int rowCount, TensorRef destination,
            int destinationRowStart) {
        Preconditions.checkArgument(layer >= 0 && layer < layers, "layer out of bounds");
        Preconditions.checkArgument(keyOrValue == 0 || keyOrValue == 1, "keyOrValue must be 0 or 1");
        Preconditions.checkArgument(blockRowStart >= 0 && rowCount >= 0 && blockRowStart + rowCount <= tokenCount,
                "block row range out of bounds");
        Preconditions.checkArgument(destination.dims() == 2 && destination.shape().last() == kvLength,
                "destination must have kvLength columns");
        Preconditions.checkArgument(destinationRowStart >= 0
                && destinationRowStart + rowCount <= destination.shape().first(), "destination row range out of bounds");
    }

    @Override
    public void close() {
        keyStorage.close();
        if (valueStorage != keyStorage) {
            valueStorage.close();
        }
    }
}
