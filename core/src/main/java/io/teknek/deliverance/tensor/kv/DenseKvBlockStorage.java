package io.teknek.deliverance.tensor.kv;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;

final class DenseKvBlockStorage implements KvBlockStorage {
    private final int layers;
    private final int tokenCount;
    private final int blockSize;
    private final int kvLength;
    private final AbstractTensor keyStorage;
    private final AbstractTensor valueStorage;

    DenseKvBlockStorage(int layers, int tokenCount, int blockSize, int kvLength, AbstractTensor keyStorage,
            AbstractTensor valueStorage) {
        this.layers = layers;
        this.tokenCount = tokenCount;
        this.blockSize = blockSize;
        this.kvLength = kvLength;
        this.keyStorage = keyStorage;
        this.valueStorage = valueStorage;
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
    public AbstractTensor rowView(int layer, int blockRow, int keyOrValue) {
        validate(layer, blockRow, keyOrValue);
        return storage(keyOrValue).slice(true, layer, blockRow);
    }

    @Override
    public AbstractTensor pageView(int layer, int keyOrValue) {
        Preconditions.checkArgument(layer >= 0 && layer < layers, "layer out of bounds");
        Preconditions.checkArgument(keyOrValue == 0 || keyOrValue == 1, "keyOrValue must be 0 or 1");
        return storage(keyOrValue).slice(true, layer);
    }

    @Override
    public void copyRow(int layer, int blockRow, int keyOrValue, AbstractTensor destination) {
        validate(layer, blockRow, keyOrValue);
        AbstractTensor storage = storage(keyOrValue);
        Preconditions.checkArgument(destination.dType() == storage.dType(), "destination dtype must match KV dtype");
        destination.copyFrom(storage, storage.getOffset(layer, blockRow, 0), 0, kvLength);
    }

    @Override
    public void copyRows(int layer, int keyOrValue, int blockRowStart, int rowCount, AbstractTensor destination,
            int destinationRowStart) {
        validateRange(layer, keyOrValue, blockRowStart, rowCount, destination, destinationRowStart);
        if (rowCount == 0) {
            return;
        }
        AbstractTensor storage = storage(keyOrValue);
        Preconditions.checkArgument(destination.dType() == storage.dType(), "destination dtype must match KV dtype");
        destination.copyFrom(storage, storage.getOffset(layer, blockRowStart, 0),
                destination.getOffset(destinationRowStart, 0), rowCount * kvLength);
    }

    private AbstractTensor storage(int keyOrValue) {
        return keyOrValue == 0 ? keyStorage : valueStorage;
    }

    AbstractTensor keyStorage() {
        return keyStorage;
    }

    AbstractTensor valueStorage() {
        return valueStorage;
    }

    private void validate(int layer, int blockRow, int keyOrValue) {
        Preconditions.checkArgument(layer >= 0 && layer < layers, "layer out of bounds");
        Preconditions.checkArgument(blockRow >= 0 && blockRow < tokenCount, "blockRow out of bounds");
        Preconditions.checkArgument(keyOrValue == 0 || keyOrValue == 1, "keyOrValue must be 0 or 1");
    }

    private void validateRange(int layer, int keyOrValue, int blockRowStart, int rowCount, AbstractTensor destination,
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
