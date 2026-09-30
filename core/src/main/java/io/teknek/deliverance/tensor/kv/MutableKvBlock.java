package io.teknek.deliverance.tensor.kv;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor2.TensorRefBackedTensor;

import javax.annotation.Nullable;
import java.util.BitSet;
import java.util.concurrent.atomic.AtomicBoolean;

final class MutableKvBlock implements AutoCloseable {
    private final int blockIndex;
    private final int blockSize;
    private final int layers;
    private final int kvLength;
    private final TensorRef keyStorage;
    private final TensorRef valueStorage;
    private final BitSet writtenRows;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final KvBufferCacheSettings settings;
    private final TensorAllocator allocator;
    private final MetricRegistry metricRegistry;
    private final TensorOperations conversionOperations;
    private boolean committed;
    private final Lighter lighter;

    MutableKvBlock(int blockIndex, int blockSize, int layers, int kvLength, DType dtype, TensorAllocator allocator,
            KvBufferCacheSettings settings, MetricRegistry metricRegistry) {
        this(blockIndex, blockSize, layers, kvLength, dtype, allocator, settings, metricRegistry, null, null);
    }

    MutableKvBlock(int blockIndex, int blockSize, int layers, int kvLength, DType dtype, TensorAllocator allocator,
            KvBufferCacheSettings settings, MetricRegistry metricRegistry,
            @Nullable TensorOperations conversionOperations, Lighter lighter) {
        this.blockIndex = blockIndex;
        this.blockSize = blockSize;
        this.layers = layers;
        this.kvLength = kvLength;
        this.allocator = allocator;
        this.settings = settings;
        this.metricRegistry = metricRegistry;
        this.conversionOperations = conversionOperations;
        this.writtenRows = new BitSet(layers * blockSize * 2);
        this.lighter = lighter;
        this.keyStorage = lighter.allocate(settings.getKvKeyDType(), TensorShape.of(layers, blockSize, kvLength));
        this.valueStorage = lighter.allocate(settings.getKvValueDType(), TensorShape.of(layers, blockSize, kvLength));
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

    void write(int layer, int position, TensorRef key, TensorRef value) {
        requireWritable();
        validateLayer(layer);
        Preconditions.checkArgument(containsPosition(position), "position not in mutable block");
        validateRow(key, "key");
        validateRow(value, "value");
        int blockRow = position - startPosition();
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, "kvcache.v2.write.key").time()) {
            copyRowIntoStorage(key, keyStorage, layer, blockRow, "key");
        }
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, "kvcache.v2.write.value").time()) {
            copyRowIntoStorage(value, valueStorage, layer, blockRow, "value");
        }
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, "kvcache.v2.write.bookkeeping").time()) {
            writtenRows.set(writtenIndex(layer, blockRow, 0));
            writtenRows.set(writtenIndex(layer, blockRow, 1));
        }
    }

    TensorRef keyRowCopy(int layer, int position, TensorAllocator allocator) {
        return rowCopy(layer, position, 0, allocator);
    }

    TensorRef valueRowCopy(int layer, int position, TensorAllocator allocator) {
        return rowCopy(layer, position, 1, allocator);
    }

    TensorRef keyRowView(int layer, int position) {
        return rowView(layer, position, 0);
    }

    TensorRef valueRowView(int layer, int position) {
        return rowView(layer, position, 1);
    }

    TensorRef keyPageView(int layer) {
        return pageView(layer, 0);
    }

    TensorRef valuePageView(int layer) {
        return pageView(layer, 1);
    }

    void copyKeyRows(int layer, int positionStart, int rowCount, TensorRef destination, int destinationRowStart) {
        copyRows(layer, positionStart, rowCount, 0, destination, destinationRowStart);
    }

    void copyValueRows(int layer, int positionStart, int rowCount, TensorRef destination, int destinationRowStart) {
        copyRows(layer, positionStart, rowCount, 1, destination, destinationRowStart);
    }

    KvBlock commit(int tokenCount) {
        requireWritable();
        Preconditions.checkArgument(tokenCount >= 0 && tokenCount <= blockSize, "tokenCount out of bounds");
        committed = true;
        KvBlockStorage blockStorage = switch (settings.getKvBlockStoragePolicy()) {
            case DENSE -> new DenseKvBlockStorage(layers, tokenCount, blockSize, kvLength,
                    keyStorage, valueStorage, lighter);
            case MSE_TURBOQUANT -> tokenCount == blockSize
                    ? MseTurboQuantKvBlockStorage.encode(combinedStorageForTurboQuant(), layers, tokenCount, blockSize, kvLength,
                    settings.getKvTurboQuantBits(), allocator, metricRegistry)
                    : new DenseKvBlockStorage(layers, tokenCount, blockSize, kvLength,
                            keyStorage, valueStorage, lighter);
        };
        return new KvBlock(blockIndex, blockSize, tokenCount, layers, kvLength, blockStorage);
    }

    private TensorRef rowCopy(int layer, int position, int keyOrValue, TensorAllocator allocator) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(containsPosition(position), "position not in mutable block");
        int blockRow = position - startPosition();
        Preconditions.checkState(writtenRows.get(writtenIndex(layer, blockRow, keyOrValue)),
                "KV row has not been written");
        TensorRef storage = storage(keyOrValue);
        TensorRef copy = lighter.allocate(storage.dType(), TensorShape.of(1, kvLength));
        lighter.copy(storage, storage.shape().getOffset(layer, blockRow, 0), copy, 0, kvLength);
        return copy;
    }

    private TensorRef rowView(int layer, int position, int keyOrValue) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(containsPosition(position), "position not in mutable block");
        int blockRow = position - startPosition();
        Preconditions.checkState(writtenRows.get(writtenIndex(layer, blockRow, keyOrValue)),
                "KV row has not been written");
        return storage(keyOrValue).slice(layer, blockRow);
    }

    private TensorRef pageView(int layer, int keyOrValue) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(keyOrValue == 0 || keyOrValue == 1, "keyOrValue must be 0 or 1");
        return storage(keyOrValue).slice(layer);
    }

    private void copyRows(int layer, int positionStart, int rowCount, int keyOrValue, TensorRef destination,
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

        TensorRef storage = storage(keyOrValue);
        Preconditions.checkArgument(destination.dType() == storage.dType(), "destination dtype must match KV dtype");

        lighter.copy(storage, storage.shape().getOffset(layer, blockRowStart, 0), destination,
                destination.shape().getOffset(destinationRowStart, 0), rowCount * kvLength);
    }

    private TensorRef storage(int keyOrValue) {
        return keyOrValue == 0 ? keyStorage : valueStorage;
    }

    private void copyRowIntoStorage(TensorRef source, TensorRef destinationStorage, int layer, int blockRow,
            String name) {
        if (source.dType() == destinationStorage.dType()) {
            try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                    "kvcache.v2.write.copy.same_dtype." + destinationStorage.dType()).time()) {
                lighter.copy(source, 0, destinationStorage, destinationStorage.shape().getOffset(layer, blockRow, 0),
                        kvLength);
            }
            return;
        }
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry,
                "kvcache.v2.write.quantize.to_" + destinationStorage.dType()).time()) {
            InferenceProfiler.counter(metricRegistry, "kvcache.v2.write.quantize.lighter").inc();
            try (TensorRef destinationRow = destinationStorage.slice(layer, blockRow)) {
                lighter.reshape(source, destinationRow);
            }
        }
    }

    private TensorRefBackedTensor combinedStorageForTurboQuant() {
        Preconditions.checkArgument(keyStorage.dType() == DType.F32 && valueStorage.dType() == DType.F32,
                "TurboQuant KV requires F32 key/value dense rows before compression");
        TensorRef combined = lighter.allocate(DType.F32, TensorShape.of(layers, 2, blockSize, kvLength));
        for (int layer = 0; layer < layers; layer++) {
            for (int blockRow = 0; blockRow < blockSize; blockRow++) {
                lighter.copy(keyStorage, keyStorage.shape().getOffset(layer, blockRow, 0), combined,
                        combined.shape().getOffset(layer, 0, blockRow, 0), kvLength);
                lighter.copy(valueStorage, valueStorage.shape().getOffset(layer, blockRow, 0), combined,
                        combined.shape().getOffset(layer, 1, blockRow, 0), kvLength);
            }
        }
        keyStorage.close();
        valueStorage.close();
        return new TensorRefBackedTensor(combined);
    }

    private int writtenIndex(int layer, int blockRow, int keyOrValue) {
        return ((layer * blockSize) + blockRow) * 2 + keyOrValue;
    }

    private void validateLayer(int layer) {
        Preconditions.checkArgument(layer >= 0 && layer < layers, "layer out of bounds");
    }

    private void validateRow(TensorRef row, String name) {
        Preconditions.checkArgument(row.dims() == 2 && row.shape().first() == 1 && row.shape().last() == kvLength,
                name + " must be [1, kvLength]");
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
            keyStorage.close();
            valueStorage.close();
        }
    }
}
