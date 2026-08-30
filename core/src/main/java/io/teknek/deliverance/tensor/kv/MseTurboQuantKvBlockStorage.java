package io.teknek.deliverance.tensor.kv;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.MseTurboQuantCodec;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;

import java.util.concurrent.TimeUnit;

final class MseTurboQuantKvBlockStorage implements KvBlockStorage {
    private static final String METRIC_PREFIX = "kvcache.v2.turboquant";

    private final DType dtype;
    private final int layers;
    private final int tokenCount;
    private final int blockSize;
    private final int kvLength;
    private final TensorAllocator allocator;
    private final MetricRegistry metricRegistry;
    private final MseTurboQuantCodec.EncodedRows encodedRows;

    private MseTurboQuantKvBlockStorage(DType dtype, int layers, int tokenCount, int blockSize, int kvLength,
            TensorAllocator allocator, MetricRegistry metricRegistry, MseTurboQuantCodec.EncodedRows encodedRows) {
        this.dtype = dtype;
        this.layers = layers;
        this.tokenCount = tokenCount;
        this.blockSize = blockSize;
        this.kvLength = kvLength;
        this.allocator = allocator;
        this.metricRegistry = metricRegistry;
        this.encodedRows = encodedRows;
    }

    static MseTurboQuantKvBlockStorage encode(AbstractTensor denseStorage, int layers, int tokenCount, int blockSize,
            int kvLength, int bitWidth, TensorAllocator allocator, MetricRegistry metricRegistry) {
        Preconditions.checkArgument(tokenCount == blockSize, "TurboQuant KV initially supports full committed blocks only");
        long start = System.nanoTime();
        MseTurboQuantCodec.EncodedRows encoded = MseTurboQuantCodec.allocate(layers * blockSize * 2, kvLength, bitWidth);
        MseTurboQuantCodec.Scratch scratch = new MseTurboQuantCodec.Scratch(encoded.rotatedDim());
        int rowIndex = 0;
        for (int layer = 0; layer < layers; layer++) {
            for (int blockRow = 0; blockRow < blockSize; blockRow++) {
                for (int keyOrValue = 0; keyOrValue < 2; keyOrValue++) {
                    try (AbstractTensor row = denseStorage.slice(true, layer, keyOrValue, blockRow)) {
                        rowIndex = MseTurboQuantCodec.encodeRow(row, encoded, rowIndex, metricRegistry, scratch,
                                METRIC_PREFIX + ".block.encode");
                    }
                }
            }
        }
        InferenceProfiler.timer(metricRegistry, METRIC_PREFIX + ".block.encode")
                .update(System.nanoTime() - start, TimeUnit.NANOSECONDS);
        long denseBytes = (long) layers * tokenCount * 2 * kvLength * denseStorage.dType().size();
        InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".dense.bytes.equivalent").inc(denseBytes);
        InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".encoded.bytes").inc(encoded.encodedBytes());
        denseStorage.close();
        return new MseTurboQuantKvBlockStorage(denseStorage.dType(), layers, tokenCount, blockSize, kvLength, allocator,
                metricRegistry, encoded);
    }

    @Override
    public KvBlockLayout layout() {
        return KvBlockLayout.MSE_TURBOQUANT;
    }

    @Override
    public DType dtype() {
        return dtype;
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
        return (long) layers * tokenCount * 2 * kvLength * dtype.size();
    }

    @Override
    public long encodedBytes() {
        return encodedRows.encodedBytes();
    }

    @Override
    public AbstractTensor rowView(int layer, int blockRow, int keyOrValue) {
        AbstractTensor decoded = allocator.getDirty(dtype, TensorShape.of(1, kvLength));
        try {
            decodeRow(layer, blockRow, keyOrValue, decoded);
            return decoded;
        } catch (RuntimeException | Error e) {
            decoded.close();
            throw e;
        }
    }

    @Override
    public AbstractTensor pageView(int layer, int keyOrValue) {
        AbstractTensor decoded = allocator.getDirty(dtype, TensorShape.of(tokenCount, kvLength));
        try {
            copyRows(layer, keyOrValue, 0, tokenCount, decoded, 0);
            return decoded;
        } catch (RuntimeException | Error e) {
            decoded.close();
            throw e;
        }
    }

    @Override
    public void copyRow(int layer, int blockRow, int keyOrValue, AbstractTensor destination) {
        decodeRow(layer, blockRow, keyOrValue, destination);
    }

    @Override
    public void copyRows(int layer, int keyOrValue, int blockRowStart, int rowCount, AbstractTensor destination,
            int destinationRowStart) {
        validateRange(layer, keyOrValue, blockRowStart, rowCount, destination, destinationRowStart);
        if (rowCount == 0) {
            return;
        }
        MseTurboQuantCodec.Scratch scratch = new MseTurboQuantCodec.Scratch(encodedRows.rotatedDim());
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, METRIC_PREFIX + ".block.decode.rows").time()) {
            for (int i = 0; i < rowCount; i++) {
                try (AbstractTensor row = destination.slice(destinationRowStart + i)) {
                    MseTurboQuantCodec.decodeRow(encodedRows, row, rowIndex(layer, blockRowStart + i, keyOrValue),
                            null, scratch, METRIC_PREFIX + ".block.decode");
                }
            }
        }
        InferenceProfiler.counter(metricRegistry, METRIC_PREFIX + ".block.decode.rows.count").inc(rowCount);
    }

    private void decodeRow(int layer, int blockRow, int keyOrValue, AbstractTensor destination) {
        validate(layer, blockRow, keyOrValue);
        Preconditions.checkArgument(destination.dims() == 2 && destination.shape().first() == 1
                && destination.shape().last() == kvLength, "destination must be [1, kvLength]");
        Preconditions.checkArgument(destination.dType() == dtype, "destination dtype must match KV dtype");
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, METRIC_PREFIX + ".block.decode.row").time()) {
            MseTurboQuantCodec.decodeRow(encodedRows, destination, rowIndex(layer, blockRow, keyOrValue), metricRegistry,
                    new MseTurboQuantCodec.Scratch(encodedRows.rotatedDim()), METRIC_PREFIX + ".block.decode");
        }
    }

    private int rowIndex(int layer, int blockRow, int keyOrValue) {
        return ((layer * blockSize) + blockRow) * 2 + keyOrValue;
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
        Preconditions.checkArgument(destination.dType() == dtype, "destination dtype must match KV dtype");
    }

    @Override
    public void close() {
        // Encoded rows are JVM-owned arrays.
    }
}
