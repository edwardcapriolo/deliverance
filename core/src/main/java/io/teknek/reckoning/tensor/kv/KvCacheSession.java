package io.teknek.reckoning.tensor.kv;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.model.InferenceProfiler;
import io.teknek.reckoning.tensor.AbstractTensor;
import io.teknek.reckoning.tensor.KvBufferCacheSettings;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.NavigableMap;
import java.util.Objects;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/** Request-local KV cache v2 session with immutable committed blocks and mutable active blocks. */
public final class KvCacheSession implements AutoCloseable {
    private static final AtomicLong SESSION_SEQUENCE = new AtomicLong();

    private final String sessionId;
    private final int layers;
    private final int contextLength;
    private final int kvLength;
    private final int blockSize;
    private final DType dtype;
    private final MetricRegistry metricRegistry;
    private final boolean trackReadViews;
    private final KvBufferCacheSettings settings;
    private final Lighter lighter;
    private final int[] layerKvLengths;
    private final NavigableMap<Integer, KvBlock> committedBlocks = new TreeMap<>();
    private final NavigableMap<Integer, KvBlockLease> committedBlockLeases = new TreeMap<>();
    private final NavigableMap<Integer, MutableKvBlock> mutableBlocks = new TreeMap<>();
    private final HashMap<Class<?>, AutoCloseable> attachments = new HashMap<>();
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private int length;

    KvCacheSession(int layers, int contextLength, int kvLength, int blockSize, DType dtype,
            MetricRegistry metricRegistry, boolean trackReadViews,
            KvBufferCacheSettings settings, Lighter lighter,
            int[] layerKvLengths) {
        this.sessionId = "kv2-session-" + SESSION_SEQUENCE.incrementAndGet();
        this.layers = layers;
        this.contextLength = contextLength;
        this.kvLength = kvLength;
        this.blockSize = blockSize;
        this.dtype = dtype;
        this.metricRegistry = metricRegistry;
        this.trackReadViews = trackReadViews;
        this.settings = settings;
        this.lighter = Objects.requireNonNull(lighter, "lighter");
        this.layerKvLengths = layerKvLengths == null ? null : layerKvLengths.clone();
    }

    KvCacheSession(int layers, int contextLength, int kvLength, int blockSize, DType dtype,
            MetricRegistry metricRegistry, boolean trackReadViews, KvBufferCacheSettings settings,
            Lighter lighter) {
        this(layers, contextLength, kvLength, blockSize, dtype, metricRegistry, trackReadViews,
                settings, lighter, null);
    }

    public int length() {
        return length;
    }

    public String sessionId() {
        return sessionId;
    }

    public int blockSize() {
        return blockSize;
    }

    int layerKvLength(int layer) {
        return layerKvLengths == null ? kvLength : layerKvLengths[layer];
    }

    public boolean isHeterogeneous() {
        return layerKvLengths != null;
    }

    DType keyDType() {
        return settings.getKvKeyDType();
    }

    DType valueDType() {
        return settings.getKvValueDType();
    }

    public List<KvBlock> committedBlocks() {
        return List.copyOf(committedBlocks.values());
    }

    /** Returns the request-local attachment of {@code type}, or {@code null} when absent. */
    public <T extends AutoCloseable> T attachment(Class<T> type) {
        requireOpen();
        return type.cast(attachments.get(Objects.requireNonNull(type, "type")));
    }

    /** Installs a request-local attachment. Replacing one closes the previous value. */
    public <T extends AutoCloseable> T attach(Class<T> type, T attachment) {
        requireOpen();
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(attachment, "attachment");
        AutoCloseable previous = attachments.put(type, attachment);
        if (previous != null) {
            closeAttachment(previous);
        }
        return attachment;
    }

    public <T extends AutoCloseable> T getOrAttach(Class<T> type, java.util.function.Supplier<T> factory) {
        requireOpen();
        T current = attachment(type);
        return current != null ? current : attach(type, factory.get());
    }

    List<KvBlockLease> committedBlockLeases() {
        return List.copyOf(committedBlockLeases.values());
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

    TensorRef keyRowCopyRef(int layer, int position) {
        return rowCopyRef(layer, position, true);
    }

    TensorRef valueRowCopyRef(int layer, int position) {
        return rowCopyRef(layer, position, false);
    }

    void write(CacheExecutionMode mode, int layer, int position, TensorRef key, TensorRef value) {
        requireOpen();
        Preconditions.checkArgument(mode != CacheExecutionMode.DENOISE_BLOCK_NO_UPDATE
                        && mode != CacheExecutionMode.READ_PREFIX_NO_UPDATE,
                mode + " must not write KV rows");
        validateLayer(layer);
        validatePosition(position);
        int blockIndex = position / blockSize;
        Preconditions.checkArgument(!committedBlocks.containsKey(blockIndex), "cannot overwrite committed KV block");
        TensorRef keyToWrite = key;
        TensorRef valueToWrite = value;
        TensorRef convertedKey = null;
        TensorRef convertedValue = null;
        try {
            if (key.dType() != settings.getKvKeyDType()) {
                convertedKey = lighter.reshape(key, settings.getKvKeyDType());
                keyToWrite = convertedKey;
            }
            if (value.dType() != settings.getKvValueDType()) {
                convertedValue = lighter.reshape(value, settings.getKvValueDType());
                valueToWrite = convertedValue;
            }
            mutableBlock(blockIndex).write(layer, position, keyToWrite, valueToWrite);
            metricRegistry.meter("kvcache.v2.row.write").mark();
        } finally {
            if (convertedKey != null) convertedKey.close();
            if (convertedValue != null) convertedValue.close();
        }
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

        ArrayList<Integer> committedToRemove = new ArrayList<>(committedBlocks.tailMap(lastBlockToKeep + 1, true).keySet());
        for (Integer blockIndex : committedToRemove) {
            closeCommittedBlock(committedBlocks.get(blockIndex), committedBlockLeases.get(blockIndex));
        }
        committedBlocks.tailMap(lastBlockToKeep + 1, true).clear();
        committedBlockLeases.tailMap(lastBlockToKeep + 1, true).clear();
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
        KvBlockLease lease = committedBlockLeases.remove(blockIndex);
        MutableKvBlock mutable = new MutableKvBlock(blockIndex, blockSize, layers, kvLength, dtype,
                settings, metricRegistry, lighter, layerKvLengths);
        try {
            for (int position = committed.startPosition(); position < newLength; position++) {
                for (int layer = 0; layer < layers; layer++) {
                        try (TensorRef key = committed.keyRowCopy(layer, position, lighter);
                             TensorRef value = committed.valueRowCopy(layer, position, lighter)) {
                        mutable.write(layer, position, key, value);
                    }
                }
            }
            closeCommittedBlock(committed, lease);
            MutableKvBlock previous = mutableBlocks.put(blockIndex, mutable);
            if (previous != null) {
                previous.close();
            }
        } catch (RuntimeException | Error e) {
            mutable.close();
            closeCommittedBlock(committed, lease);
            throw e;
        }
    }

    /** Transfers an owned committed block out of this session for shared-block admission. */
    public KvBlock detachCommittedBlock(int blockIndex) {
        requireOpen();
        KvBlockLease lease = committedBlockLeases.remove(blockIndex);
        Preconditions.checkState(lease == null, "cannot detach a leased KV block");
        KvBlock block = committedBlocks.remove(blockIndex);
        Preconditions.checkState(block != null, "no committed KV block at index " + blockIndex);
        return block;
    }

    /** Attaches a retained shared immutable block lease to this session. */
    public void attachCommittedBlock(KvBlockLease lease) {
        requireOpen();
        Objects.requireNonNull(lease, "lease");
        KvBlock block = lease.block();
        Preconditions.checkArgument(block.tokenCount() == blockSize,
                "shared KV block must be a full committed block");
        Preconditions.checkArgument(block.blockSize() == blockSize, "block size mismatch");
        Preconditions.checkArgument(block.endPositionExclusive() <= length || block.startPosition() == length,
                "shared KV blocks must attach contiguously or inside current length");
        Preconditions.checkArgument(!mutableBlocks.containsKey(block.blockIndex()),
                "cannot attach shared block over mutable block");
        Preconditions.checkArgument(!committedBlocks.containsKey(block.blockIndex()),
                "cannot attach shared block over committed block");
        committedBlocks.put(block.blockIndex(), block);
        committedBlockLeases.put(block.blockIndex(), lease);
        length = Math.max(length, block.endPositionExclusive());
        metricRegistry.meter("kvcache.v2.block.attach").mark();
    }

    List<Integer> attachedLeaseBlockIndexes() {
        return Collections.unmodifiableList(new ArrayList<>(committedBlockLeases.keySet()));
    }

    public boolean committedBlockIsLeased(int blockIndex) {
        return committedBlockLeases.containsKey(blockIndex);
    }

    TensorRef copyVisibleKeysRef(int layer, int visibleTokens) {
        return copyVisible(layer, visibleTokens, true);
    }

    TensorRef copyVisibleValuesRef(int layer, int visibleTokens) {
        return copyVisible(layer, visibleTokens, false);
    }

    void copyKeyRows(int layer, int positionStart, int rowCount, TensorRef destination, int destinationRowStart) {
        copyRows(layer, positionStart, rowCount, destination, destinationRowStart, true);
    }

    void copyValueRows(int layer, int positionStart, int rowCount, TensorRef destination, int destinationRowStart) {
        copyRows(layer, positionStart, rowCount, destination, destinationRowStart, false);
    }

    TensorRef[] keyPageRefs(int layer, int visibleTokens) {
        return pages(layer, visibleTokens, true);
    }

    TensorRef[] valuePageRefs(int layer, int visibleTokens) {
        return pages(layer, visibleTokens, false);
    }

    private TensorRef copyVisible(int layer, int visibleTokens, boolean key) {
        TensorRef result = lighter.allocateDirty(dtype,
                io.teknek.reckoning.tensor.TensorShape.of(visibleTokens, layerKvLength(layer)));
        try {
            for (int position = 0; position < visibleTokens; position++) {
                try (TensorRef row = rowCopyRef(layer, position, key)) {
                    lighter.copyFrom(row, result, 0, result.shape().getOffset(position, 0), layerKvLength(layer));
                }
            }
            return result;
        } catch (RuntimeException | Error e) {
            result.close();
            throw e;
        }
    }

    private void copyRows(int layer, int positionStart, int rowCount, TensorRef destination, int destinationRowStart,
            boolean key) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(positionStart >= 0 && rowCount >= 0 && positionStart + rowCount <= length,
                "position range out of visible length");
        Preconditions.checkArgument(destination.dims() == 2 && destination.shape().last() == layerKvLength(layer),
                "destination must have layer KV width columns");
        Preconditions.checkArgument(destinationRowStart >= 0
                && destinationRowStart + rowCount <= destination.shape().first(), "destination row range out of bounds");
        int remaining = rowCount;
        int sourcePosition = positionStart;
        int destRow = destinationRowStart;
        while (remaining > 0) {
            int blockIndex = sourcePosition / blockSize;
            int blockRow = sourcePosition % blockSize;
            int rowsInBlock = Math.min(remaining, blockSize - blockRow);
            KvBlock committed = committedBlocks.get(blockIndex);
            if (committed != null) {
                rowsInBlock = Math.min(rowsInBlock, committed.endPositionExclusive() - sourcePosition);
                if (key) {
                    committed.copyKeyRows(layer, sourcePosition, rowsInBlock, destination, destRow);
                } else {
                    committed.copyValueRows(layer, sourcePosition, rowsInBlock, destination, destRow);
                }
            } else {
                MutableKvBlock mutable = mutableBlocks.get(blockIndex);
                if (mutable == null) {
                    throw new IllegalStateException("No KV block for position " + sourcePosition);
                }
                if (key) {
                    mutable.copyKeyRows(layer, sourcePosition, rowsInBlock, destination, destRow);
                } else {
                    mutable.copyValueRows(layer, sourcePosition, rowsInBlock, destination, destRow);
                }
            }
            remaining -= rowsInBlock;
            sourcePosition += rowsInBlock;
            destRow += rowsInBlock;
        }
    }

    private TensorRef[] pages(int layer, int visibleTokens, boolean key) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(visibleTokens >= 0 && visibleTokens <= length,
                "visibleTokens must be within session length");
        if (visibleTokens == 0) {
            return new TensorRef[0];
        }
        int pageCount = (visibleTokens + blockSize - 1) / blockSize;
        // TensorRef-backed pages are caller-owned views and are closed after each attention pass.
        return buildPages(layer, pageCount, key);
    }

    private TensorRef[] buildPages(int layer, int pageCount, boolean key) {
        TensorRef[] pages = new TensorRef[pageCount];
        try {
            for (int blockIndex = 0; blockIndex < pageCount; blockIndex++) {
                KvBlock committed = committedBlocks.get(blockIndex);
                if (committed != null) {
                    pages[blockIndex] = key ? committed.keyPageView(layer) : committed.valuePageView(layer);
                } else {
                    MutableKvBlock mutable = mutableBlocks.get(blockIndex);
                    if (mutable == null) {
                        throw new IllegalStateException("No KV block for block index " + blockIndex);
                    }
                    pages[blockIndex] = key ? mutable.keyPageView(layer) : mutable.valuePageView(layer);
                }
            }
            return pages;
        } catch (RuntimeException | Error e) {
            for (TensorRef page : pages) {
                if (page != null) {
                    page.close();
                }
            }
            throw e;
        }
    }

    private TensorRef rowCopyRef(int layer, int position, boolean key) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(position >= 0 && position < length, "position out of visible length");
        int blockIndex = position / blockSize;
        KvBlock committed = committedBlocks.get(blockIndex);
        if (committed != null) {
            return key ? committed.keyRowCopy(layer, position, lighter) : committed.valueRowCopy(layer, position, lighter);
        }
        MutableKvBlock mutable = mutableBlocks.get(blockIndex);
        if (mutable == null) {
            throw new IllegalStateException("No KV block for position " + position);
        }
        return key ? mutable.keyRowCopy(layer, position, lighter) : mutable.valueRowCopy(layer, position, lighter);
    }

    TensorRef keyRowViewRef(int layer, int position) {
        return rowViewRef(layer, position, true);
    }

    TensorRef valueRowViewRef(int layer, int position) {
        return rowViewRef(layer, position, false);
    }

    private TensorRef rowViewRef(int layer, int position, boolean key) {
        requireOpen();
        validateLayer(layer);
        Preconditions.checkArgument(position >= 0 && position < length, "position out of visible length");
        if (isHeterogeneous()) {
            return rowCopyRef(layer, position, key);
        }
        int blockIndex = position / blockSize;
        TensorRef view;
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
        return view;
    }

    void copyKeyRows(int layer, int positionStart, int rowCount, AbstractTensor destination, int destinationRowStart) {
        try (TensorRef destinationRef = TensorRef.borrowed(destination)) {
            copyKeyRows(layer, positionStart, rowCount, destinationRef, destinationRowStart);
        }
    }

    void copyValueRows(int layer, int positionStart, int rowCount, AbstractTensor destination, int destinationRowStart) {
        try (TensorRef destinationRef = TensorRef.borrowed(destination)) {
            copyValueRows(layer, positionStart, rowCount, destinationRef, destinationRowStart);
        }
    }

    private MutableKvBlock mutableBlock(int blockIndex) {
        return mutableBlocks.computeIfAbsent(blockIndex,
                index -> new MutableKvBlock(index, blockSize, layers, kvLength, dtype, settings,
                        metricRegistry, lighter, layerKvLengths));
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

    private void closeCommittedBlock(KvBlock block, KvBlockLease lease) {
        if (lease != null) {
            lease.close();
        } else {
            block.close();
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
            for (var entry : committedBlocks.entrySet()) {
                closeCommittedBlock(entry.getValue(), committedBlockLeases.get(entry.getKey()));
            }
            mutableBlocks.values().forEach(MutableKvBlock::close);
            committedBlocks.clear();
            committedBlockLeases.clear();
            mutableBlocks.clear();
            attachments.values().forEach(this::closeAttachment);
            attachments.clear();
            metricRegistry.meter("kvcache.v2.session.close").mark();
        }
    }

    private void closeAttachment(AutoCloseable attachment) {
        try {
            attachment.close();
        } catch (Exception e) {
            throw new IllegalStateException("Unable to close KV session attachment", e);
        }
    }
}
