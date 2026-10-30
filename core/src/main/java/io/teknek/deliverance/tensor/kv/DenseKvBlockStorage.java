package io.teknek.deliverance.tensor.kv;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Arrays;

final class DenseKvBlockStorage implements KvBlockStorage {
    private final int layers;
    private final int tokenCount;
    private final int blockSize;
    private final int kvLength;
    private final TensorRef[] keyStorage;
    private final TensorRef[] valueStorage;
    private final int[] layerKvLengths;
    private final Lighter lighter;

    DenseKvBlockStorage(int layers, int tokenCount, int blockSize, int kvLength, TensorRef keyStorage,
            TensorRef valueStorage, Lighter lighter) {
        this(layers, tokenCount, blockSize, kvLength, repeated(keyStorage, layers), repeated(valueStorage, layers),
                null, lighter);
    }

    DenseKvBlockStorage(int layers, int tokenCount, int blockSize, int kvLength, TensorRef[] keyStorage,
            TensorRef[] valueStorage, int[] layerKvLengths, Lighter lighter) {
        this.layers = layers;
        this.tokenCount = tokenCount;
        this.blockSize = blockSize;
        this.kvLength = kvLength;
        this.keyStorage = keyStorage;
        this.valueStorage = valueStorage;
        this.layerKvLengths = layerKvLengths == null ? null : layerKvLengths.clone();
        this.lighter = lighter;
    }

    @Override
    public KvBlockLayout layout() {
        return KvBlockLayout.DENSE;
    }

    @Override
    public DType dtype() {
        return keyStorage[0].dType();
    }

    @Override
    public DType dtype(int keyOrValue) {
        return storage(keyOrValue, 0).dType();
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
    public int layerKvLength(int layer) {
        Preconditions.checkArgument(layer >= 0 && layer < layers, "layer out of bounds");
        return layerKvLengths == null ? kvLength : layerKvLengths[layer];
    }

    @Override
    public long denseBytesEquivalent() {
        long bytes = 0;
        for (int layer = 0; layer < layers; layer++) {
            bytes += (long) tokenCount * layerKvLength(layer) * keyStorage[layer].dType().size();
            bytes += (long) tokenCount * layerKvLength(layer) * valueStorage[layer].dType().size();
        }
        return bytes;
    }

    @Override
    public long encodedBytes() {
        return denseBytesEquivalent();
    }

    @Override
    public TensorRef rowView(int layer, int blockRow, int keyOrValue) {
        validate(layer, blockRow, keyOrValue);
        return storage(keyOrValue, layer).slice(blockRow);
    }

    @Override
    public TensorRef pageView(int layer, int keyOrValue) {
        Preconditions.checkArgument(layer >= 0 && layer < layers, "layer out of bounds");
        Preconditions.checkArgument(keyOrValue == 0 || keyOrValue == 1, "keyOrValue must be 0 or 1");
        return storage(keyOrValue, layer).slice();
    }

    @Override
    public void copyRow(int layer, int blockRow, int keyOrValue, TensorRef destination) {
        validate(layer, blockRow, keyOrValue);
        TensorRef storage = storage(keyOrValue, layer);
        Preconditions.checkArgument(destination.dType() == storage.dType(), "destination dtype must match KV dtype");
        int width = layerKvLength(layer);
        Preconditions.checkArgument(destination.shape().last() == width, "destination must have layer KV width columns");
        lighter.copy(storage, storage.shape().getOffset(blockRow, 0), destination, 0, width);
    }

    @Override
    public void copyRows(int layer, int keyOrValue, int blockRowStart, int rowCount, TensorRef destination,
            int destinationRowStart) {
        validateRange(layer, keyOrValue, blockRowStart, rowCount, destination, destinationRowStart);
        if (rowCount == 0) {
            return;
        }
        TensorRef storage = storage(keyOrValue, layer);
        Preconditions.checkArgument(destination.dType() == storage.dType(), "destination dtype must match KV dtype");
        int width = layerKvLength(layer);
        Preconditions.checkArgument(destination.shape().last() == width, "destination must have layer KV width columns");
        lighter.copy(storage, storage.shape().getOffset(blockRowStart, 0), destination,
                destination.shape().getOffset(destinationRowStart, 0), rowCount * width);
    }

    private TensorRef storage(int keyOrValue, int layer) {
        return (keyOrValue == 0 ? keyStorage : valueStorage)[layer];
    }

    TensorRef keyStorage() {
        return keyStorage.length == 1 ? keyStorage[0] : keyStorage[0];
    }

    TensorRef valueStorage() {
        return valueStorage[0];
    }

    boolean hasUniformLayerWidths() {
        for (int layer = 1; layer < layers; layer++) {
            if (layerKvLength(layer) != layerKvLength(0)) return false;
        }
        return true;
    }

    private static TensorRef[] repeated(TensorRef tensor, int count) {
        TensorRef[] result = new TensorRef[count];
        Arrays.fill(result, tensor);
        return result;
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
        Preconditions.checkArgument(destination.dims() == 2 && destination.shape().last() == layerKvLength(layer),
                "destination must have layer KV width columns");
        Preconditions.checkArgument(destinationRowStart >= 0
                && destinationRowStart + rowCount <= destination.shape().first(), "destination row range out of bounds");
    }

    @Override
    public void close() {
        keyStorage[0].close();
        for (TensorRef storage : keyStorage) if (storage != keyStorage[0]) storage.close();
        for (TensorRef storage : valueStorage) if (storage != keyStorage[0]) storage.close();
    }
}
