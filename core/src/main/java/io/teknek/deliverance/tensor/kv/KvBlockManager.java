package io.teknek.deliverance.tensor.kv;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;

import javax.annotation.Nullable;
import java.util.Comparator;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/** Concurrent owner of reusable immutable KVCache2 blocks. */
public final class KvBlockManager implements AutoCloseable {
    private enum State {
        AVAILABLE,
        EVICTING,
        CLOSED
    }

    static final class ManagedKvBlock {
        private final KvBlockManager owner;
        private final KvBlockKey key;
        private final KvBlock block;
        private final MetricRegistry metricRegistry;
        private final AtomicInteger refCount = new AtomicInteger();
        private final AtomicLong lastAccessNanos = new AtomicLong(System.nanoTime());
        private volatile State state = State.AVAILABLE;

        private ManagedKvBlock(KvBlockManager owner, KvBlockKey key, KvBlock block, MetricRegistry metricRegistry) {
            this.owner = owner;
            this.key = key;
            this.block = block;
            this.metricRegistry = metricRegistry;
        }

        KvBlockKey key() {
            return key;
        }

        KvBlock block() {
            return block;
        }

        long encodedBytes() {
            return block.encodedBytes();
        }

        int refCount() {
            return refCount.get();
        }

        long lastAccessNanos() {
            return lastAccessNanos.get();
        }

        KvBlockLease tryRetain(String sessionId) {
            Objects.requireNonNull(sessionId, "sessionId");
            synchronized (this) {
                if (state != State.AVAILABLE) {
                    return null;
                }
                refCount.incrementAndGet();
                lastAccessNanos.set(System.nanoTime());
                InferenceProfiler.counter(metricRegistry, "kvcache.v2.blockmanager.retain").inc();
                return new KvBlockLease(this, sessionId);
            }
        }

        void markEvicting() {
            boolean closeNow;
            synchronized (this) {
                if (state != State.AVAILABLE) {
                    return;
                }
                state = State.EVICTING;
                closeNow = refCount.get() == 0;
            }
            if (closeNow) {
                closeBlock();
            }
        }

        void release() {
            int remaining = refCount.decrementAndGet();
            Preconditions.checkState(remaining >= 0, "KV block lease refcount went negative");
            InferenceProfiler.counter(metricRegistry, "kvcache.v2.blockmanager.release").inc();
            if (remaining == 0 && state == State.EVICTING) {
                closeBlock();
            }
            if (remaining == 0) {
                owner.evictToBudget();
            }
        }

        private void closeBlock() {
            boolean shouldClose;
            synchronized (this) {
                shouldClose = state != State.CLOSED;
                state = State.CLOSED;
            }
            if (shouldClose) {
                block.close();
                InferenceProfiler.counter(metricRegistry, "kvcache.v2.blockmanager.close").inc();
            }
        }
    }

    private final ConcurrentHashMap<KvBlockKey, ManagedKvBlock> blocks = new ConcurrentHashMap<>();
    private final MetricRegistry metricRegistry;
    private final long maxResidentBytes;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    public KvBlockManager(MetricRegistry metricRegistry) {
        this(metricRegistry, Long.MAX_VALUE);
    }

    public KvBlockManager(MetricRegistry metricRegistry, KvBufferCacheSettings settings) {
        this(metricRegistry, Objects.requireNonNull(settings, "settings").getSharedPrefixBlockCacheMaxBytes());
    }

    public KvBlockManager(MetricRegistry metricRegistry, long maxResidentBytes) {
        this.metricRegistry = Objects.requireNonNull(metricRegistry, "metricRegistry");
        Preconditions.checkArgument(maxResidentBytes >= 0, "maxResidentBytes must be >= 0");
        this.maxResidentBytes = maxResidentBytes;
    }

    public KvBlockLease retain(KvBlockKey key, String sessionId) {
        requireOpen();
        Objects.requireNonNull(key, "key");
        InferenceProfiler.counter(metricRegistry, "kvcache.v2.blockmanager.lookup").inc();
        while (true) {
            ManagedKvBlock managed = blocks.get(key);
            if (managed == null) {
                InferenceProfiler.counter(metricRegistry, "kvcache.v2.blockmanager.miss").inc();
                return null;
            }
            KvBlockLease lease = managed.tryRetain(sessionId);
            if (lease != null) {
                InferenceProfiler.counter(metricRegistry, "kvcache.v2.blockmanager.hit.memory").inc();
                return lease;
            }
            blocks.remove(key, managed);
            InferenceProfiler.counter(metricRegistry, "kvcache.v2.blockmanager.retain.race_retry").inc();
        }
    }

    public KvBlockLease admitAndRetain(KvBlockKey key, KvBlock candidate, String sessionId) {
        requireOpen();
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(candidate, "candidate");
        validateKeyMatchesBlock(key, candidate);
        ManagedKvBlock candidateManaged = new ManagedKvBlock(this, key, candidate, metricRegistry);
        while (true) {
            ManagedKvBlock existing = blocks.putIfAbsent(key, candidateManaged);
            if (existing == null) {
                InferenceProfiler.counter(metricRegistry, "kvcache.v2.blockmanager.admit").inc();
                KvBlockLease lease = candidateManaged.tryRetain(sessionId);
                evictToBudget();
                return lease;
            }
            KvBlockLease lease = existing.tryRetain(sessionId);
            if (lease != null) {
                InferenceProfiler.counter(metricRegistry, "kvcache.v2.blockmanager.admit.duplicate").inc();
                candidate.close();
                return lease;
            }
            blocks.remove(key, existing);
            InferenceProfiler.counter(metricRegistry, "kvcache.v2.blockmanager.retain.race_retry").inc();
        }
    }

    public boolean evict(KvBlockKey key) {
        requireOpen();
        Objects.requireNonNull(key, "key");
        ManagedKvBlock managed = blocks.remove(key);
        if (managed == null) {
            return false;
        }
        managed.markEvicting();
        InferenceProfiler.counter(metricRegistry, "kvcache.v2.blockmanager.evict").inc();
        InferenceProfiler.counter(metricRegistry, "kvcache.v2.blockmanager.evict.bytes").inc(managed.encodedBytes());
        return true;
    }

    public int residentBlockCount() {
        return blocks.size();
    }

    public int refCount(KvBlockKey key) {
        ManagedKvBlock managed = blocks.get(key);
        return managed == null ? 0 : managed.refCount();
    }

    public long residentEncodedBytes() {
        long bytes = 0;
        for (ManagedKvBlock managed : blocks.values()) {
            bytes += managed.encodedBytes();
        }
        return bytes;
    }

    public long referencedEncodedBytes() {
        long bytes = 0;
        for (ManagedKvBlock managed : blocks.values()) {
            if (managed.refCount() > 0) {
                bytes += managed.encodedBytes();
            }
        }
        return bytes;
    }

    public long maxResidentBytes() {
        return maxResidentBytes;
    }

    void evictToBudget() {
        if (maxResidentBytes == Long.MAX_VALUE || closed.get()) {
            return;
        }
        while (residentEncodedBytes() > maxResidentBytes) {
            ManagedKvBlock victim = coldestUnreferencedBlock();
            if (victim == null) {
                long excess = residentEncodedBytes() - maxResidentBytes;
                if (excess > 0) {
                    InferenceProfiler.counter(metricRegistry, "kvcache.v2.blockmanager.over_budget").inc();
                    InferenceProfiler.counter(metricRegistry, "kvcache.v2.blockmanager.over_budget.bytes").inc(excess);
                }
                return;
            }
            if (blocks.remove(victim.key(), victim)) {
                long evictedBytes = victim.encodedBytes();
                victim.markEvicting();
                InferenceProfiler.counter(metricRegistry, "kvcache.v2.blockmanager.evict").inc();
                InferenceProfiler.counter(metricRegistry, "kvcache.v2.blockmanager.evict.bytes").inc(evictedBytes);
            }
        }
    }

    @Nullable
    private ManagedKvBlock coldestUnreferencedBlock() {
        return blocks.values().stream()
                .filter(block -> block.refCount() == 0 && block.state == State.AVAILABLE)
                .min(Comparator.comparingLong(ManagedKvBlock::lastAccessNanos))
                .orElse(null);
    }

    public void clear() {
        for (KvBlockKey key : blocks.keySet()) {
            evict(key);
        }
    }

    private void validateKeyMatchesBlock(KvBlockKey key, KvBlock block) {
        Preconditions.checkArgument(key.blockIndex() == block.blockIndex(), "key blockIndex does not match block");
        Preconditions.checkArgument(key.blockSize() == block.blockSize(), "key blockSize does not match block");
        Preconditions.checkArgument(key.tokenCount() == block.tokenCount(), "key tokenCount does not match block");
        Preconditions.checkArgument(key.layout() == block.layout(), "key layout does not match block");
    }

    private void requireOpen() {
        if (closed.get()) {
            throw new IllegalStateException("KV block manager is closed");
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            for (ManagedKvBlock managed : blocks.values()) {
                managed.markEvicting();
            }
            blocks.clear();
        }
    }
}
