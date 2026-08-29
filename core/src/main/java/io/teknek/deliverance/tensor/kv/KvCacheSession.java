package io.teknek.deliverance.tensor.kv;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ReadOnlyTensor;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TrackedReadOnlyTensor;

import java.util.ArrayList;
import java.util.List;
import java.util.NavigableMap;
import java.util.Objects;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicBoolean;

/** Request-local KV cache v2 session with immutable committed blocks and mutable active blocks. */
public final class KvCacheSession implements AutoCloseable {
    private final int layers;
    private final int contextLength;
    private final int kvLength;
    private final int blockSize;
    private final DType dtype;
    private final TensorAllocator allocator;
    private final MetricRegistry metricRegistry;
    private final boolean trackReadViews;
    private final NavigableMap<Integer, KvBlock> committedBlocks = new TreeMap<>();
    private final NavigableMap<Integer, MutableKvBlock> mutableBlocks = new TreeMap<>();
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private int length;

    KvCacheSession(int layers, int contextLength, int kvLength, int blockSize, DType dtype,
            TensorAllocator allocator, MetricRegistry metricRegistry, boolean trackReadViews) {
        this.layers = layers;
        this.contextLength = contextLength;
        this.kvLength = kvLength;
        this.blockSize = blockSize;
        this.dtype = dtype;
        this.allocator = allocator;
        this.metricRegistry = metricRegistry;
        this.trackReadViews = trackReadViews;
    }

    public int length() {
        return length;
    }

    public int blockSize() {
        return blockSize;
    }

    public List<KvBlock> committedBlocks() {
        return List.copyOf(committedBlocks.values());
    }

    public KvWriteCursor writer(CacheExecutionMode mode) {
        requireOpen();
        return new KvWriteCursor(this, Objects.requireNonNull(mode, "mode"));
    }

    public KvReadView readView(int layer, int visibleTokens, AttentionPattern pattern) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(visibleTokens >= 0 && visibleTokens <= length,
                "visibleTokens must be within session length");
        return new KvReadView(this, layer, visibleTokens, Objects.requireNonNull(pattern, "pattern"));
    }

    public AbstractTensor keyRowCopy(int layer, int position) {
        return rowCopy(layer, position, true);
    }

    public AbstractTensor valueRowCopy(int layer, int position) {
        return rowCopy(layer, position, false);
    }

    AbstractTensor keyRowView(int layer, int position) {
        return rowView(layer, position, true);
    }

    AbstractTensor valueRowView(int layer, int position) {
        return rowView(layer, position, false);
    }

    void write(CacheExecutionMode mode, int layer, int position, AbstractTensor key, AbstractTensor value) {
        requireOpen();
        Preconditions.checkArgument(mode != CacheExecutionMode.DENOISE_BLOCK_NO_UPDATE
                        && mode != CacheExecutionMode.READ_PREFIX_NO_UPDATE,
                mode + " must not write KV rows");
        validateLayer(layer);
        validatePosition(position);
        int blockIndex = position / blockSize;
        Preconditions.checkArgument(!committedBlocks.containsKey(blockIndex), "cannot overwrite committed KV block");
        mutableBlock(blockIndex).write(layer, position, key, value);
        metricRegistry.meter("kvcache.v2.row.write").mark();
    }

    public void advanceLength(int newLength) {
        requireOpen();
        Preconditions.checkArgument(newLength >= length, "newLength must be >= current length");
        Preconditions.checkArgument(newLength <= contextLength, "newLength exceeds context length");
        length = newLength;
        commitFullBlocksBefore(length);
    }

    public void crop(int newLength) {
        requireOpen();
        Preconditions.checkArgument(newLength >= 0 && newLength <= length, "newLength out of bounds");
        int lastBlockToKeep = newLength == 0 ? -1 : (newLength - 1) / blockSize;
        int rowsInLastBlock = newLength == 0 ? 0 : ((newLength - 1) % blockSize) + 1;

        committedBlocks.tailMap(lastBlockToKeep + 1, true).values().forEach(KvBlock::close);
        committedBlocks.tailMap(lastBlockToKeep + 1, true).clear();
        mutableBlocks.tailMap(lastBlockToKeep + 1, true).values().forEach(MutableKvBlock::close);
        mutableBlocks.tailMap(lastBlockToKeep + 1, true).clear();

        if (rowsInLastBlock > 0 && rowsInLastBlock < blockSize) {
            splitCommittedTailBlock(lastBlockToKeep, newLength);
        }
        length = newLength;
        metricRegistry.meter("kvcache.v2.crop").mark();
    }

    private void splitCommittedTailBlock(int blockIndex, int newLength) {
        KvBlock committed = committedBlocks.remove(blockIndex);
        if (committed == null) {
            return;
        }
        MutableKvBlock mutable = new MutableKvBlock(blockIndex, blockSize, layers, kvLength, dtype, allocator);
        try {
            for (int position = committed.startPosition(); position < newLength; position++) {
                for (int layer = 0; layer < layers; layer++) {
                    try (AbstractTensor key = committed.keyRowCopy(layer, position, allocator);
                         AbstractTensor value = committed.valueRowCopy(layer, position, allocator)) {
                        mutable.write(layer, position, key, value);
                    }
                }
            }
            committed.close();
            MutableKvBlock previous = mutableBlocks.put(blockIndex, mutable);
            if (previous != null) {
                previous.close();
            }
        } catch (RuntimeException | Error e) {
            mutable.close();
            committed.close();
            throw e;
        }
    }

    AbstractTensor copyVisibleKeys(int layer, int visibleTokens) {
        return copyVisible(layer, visibleTokens, true);
    }

    AbstractTensor copyVisibleValues(int layer, int visibleTokens) {
        return copyVisible(layer, visibleTokens, false);
    }

    private AbstractTensor copyVisible(int layer, int visibleTokens, boolean key) {
        AbstractTensor result = allocator.getDirty(dtype, io.teknek.deliverance.tensor.TensorShape.of(visibleTokens, kvLength));
        try {
            for (int position = 0; position < visibleTokens; position++) {
                try (AbstractTensor row = rowCopy(layer, position, key)) {
                    result.copyFrom(row, 0, result.getOffset(position, 0), kvLength);
                }
            }
            return result;
        } catch (RuntimeException | Error e) {
            result.close();
            throw e;
        }
    }

    private AbstractTensor rowCopy(int layer, int position, boolean key) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(position >= 0 && position < length, "position out of visible length");
        int blockIndex = position / blockSize;
        KvBlock committed = committedBlocks.get(blockIndex);
        if (committed != null) {
            return key ? committed.keyRowCopy(layer, position, allocator) : committed.valueRowCopy(layer, position, allocator);
        }
        MutableKvBlock mutable = mutableBlocks.get(blockIndex);
        if (mutable == null) {
            throw new IllegalStateException("No KV block for position " + position);
        }
        return key ? mutable.keyRowCopy(layer, position, allocator) : mutable.valueRowCopy(layer, position, allocator);
    }

    private AbstractTensor rowView(int layer, int position, boolean key) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(position >= 0 && position < length, "position out of visible length");
        int blockIndex = position / blockSize;
        AbstractTensor view;
        KvBlock committed = committedBlocks.get(blockIndex);
        if (committed != null) {
            view = key ? committed.keyRowView(layer, position) : committed.valueRowView(layer, position);
        } else {
            MutableKvBlock mutable = mutableBlocks.get(blockIndex);
            if (mutable == null) {
                throw new IllegalStateException("No KV block for position " + position);
            }
            view = key ? mutable.keyRowView(layer, position) : mutable.valueRowView(layer, position);
        }
        return trackReadViews ? new TrackedReadOnlyTensor(view) : new ReadOnlyTensor(view);
    }

    private MutableKvBlock mutableBlock(int blockIndex) {
        return mutableBlocks.computeIfAbsent(blockIndex,
                index -> new MutableKvBlock(index, blockSize, layers, kvLength, dtype, allocator));
    }

    private void commitFullBlocksBefore(int newLength) {
        int fullBlocks = newLength / blockSize;
        List<Integer> toCommit = new ArrayList<>(mutableBlocks.headMap(fullBlocks, false).keySet());
        for (Integer blockIndex : toCommit) {
            MutableKvBlock mutable = mutableBlocks.remove(blockIndex);
            committedBlocks.put(blockIndex, mutable.commit(blockSize));
            metricRegistry.meter("kvcache.v2.block.commit").mark();
        }
    }

    private void validateLayer(int layer) {
        Preconditions.checkArgument(layer >= 0 && layer < layers, "layer out of bounds");
    }

    private void validatePosition(int position) {
        Preconditions.checkArgument(position >= 0 && position < contextLength, "position out of bounds");
    }

    private void requireOpen() {
        if (closed.get()) {
            throw new IllegalStateException("KV cache session is closed");
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            committedBlocks.values().forEach(KvBlock::close);
            mutableBlocks.values().forEach(MutableKvBlock::close);
            committedBlocks.clear();
            mutableBlocks.clear();
            metricRegistry.meter("kvcache.v2.session.close").mark();
        }
    }
}
