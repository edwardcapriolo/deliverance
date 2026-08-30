package io.teknek.deliverance.tensor.kv;

import com.google.common.base.Preconditions;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/** Temporary snapshot prefix cache for KVCache2 sessions. */
public final class KvPrefixSnapshotCache implements AutoCloseable {
    public record CacheKey(Optional<String> salt, List<Integer> prefixTokens) {
        public CacheKey {
            salt = salt == null ? Optional.empty() : salt;
            prefixTokens = List.copyOf(prefixTokens);
        }
    }

    public record PrefixHit(int length) {
    }

    private interface StoredPrefixEntry extends AutoCloseable {
        int length();

        void restoreTo(KvCacheSession destination);

        @Override
        void close();
    }

    private final class DenseStoredPrefixEntry implements StoredPrefixEntry {
        private final int length;
        private final AbstractTensor[] keysByLayer;
        private final AbstractTensor[] valuesByLayer;
        private final AtomicBoolean closed = new AtomicBoolean(false);

        private DenseStoredPrefixEntry(int length, AbstractTensor[] keysByLayer, AbstractTensor[] valuesByLayer) {
            this.length = length;
            this.keysByLayer = keysByLayer;
            this.valuesByLayer = valuesByLayer;
        }

        @Override
        public int length() {
            return length;
        }

        @Override
        public void restoreTo(KvCacheSession destination) {
            if (closed.get()) {
                throw new IllegalStateException("prefix snapshot is closed");
            }
            long start = System.nanoTime();
            try (KvWriteCursor writer = destination.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
                for (int position = 0; position < length; position++) {
                    for (int layer = 0; layer < layers; layer++) {
                        try (AbstractTensor key = keysByLayer[layer].slice(position);
                             AbstractTensor value = valuesByLayer[layer].slice(position)) {
                            writer.write(layer, position, key, value);
                        }
                    }
                    writer.advanceLength(position + 1);
                }
            }
            InferenceProfiler.timer(metricRegistry, "kvcache.v2.prefix.restore")
                    .update(System.nanoTime() - start, TimeUnit.NANOSECONDS);
        }

        @Override
        public void close() {
            if (closed.compareAndSet(false, true)) {
                for (AbstractTensor key : keysByLayer) {
                    key.close();
                }
                for (AbstractTensor value : valuesByLayer) {
                    value.close();
                }
            }
        }
    }

    private final int layers;
    private final int contextLength;
    private final int kvLength;
    private final int blockSize;
    private final DType keyDType;
    private final DType valueDType;
    private final TensorAllocator allocator;
    private final MetricRegistry metricRegistry;
    private final KvBufferCacheSettings settings;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final Map<CacheKey, StoredPrefixEntry> prefixCache;

    public KvPrefixSnapshotCache(int layers, int contextLength, int kvLength, int blockSize, DType keyDType,
            DType valueDType,
            TensorAllocator allocator, MetricRegistry metricRegistry, KvBufferCacheSettings settings) {
        Preconditions.checkArgument(layers > 0, "layers must be > 0");
        Preconditions.checkArgument(contextLength > 0, "contextLength must be > 0");
        Preconditions.checkArgument(kvLength > 0, "kvLength must be > 0");
        Preconditions.checkArgument(blockSize > 0, "blockSize must be > 0");
        this.layers = layers;
        this.contextLength = contextLength;
        this.kvLength = kvLength;
        this.blockSize = blockSize;
        this.keyDType = keyDType;
        this.valueDType = valueDType;
        this.allocator = allocator;
        this.metricRegistry = metricRegistry;
        this.settings = settings;
        this.prefixCache = Collections.synchronizedMap(new LinkedHashMap<CacheKey, StoredPrefixEntry>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<CacheKey, StoredPrefixEntry> eldest) {
                boolean evict = size() > KvPrefixSnapshotCache.this.settings.getMaxEntries();
                if (evict && eldest != null && eldest.getValue() != null) {
                    InferenceProfiler.counter(metricRegistry, "kvcache.v2.prefix.evict").inc();
                    eldest.getValue().close();
                }
                return evict;
            }
        });
    }

    public PrefixHit lookupPrefix(int[] tokens, Optional<String> salt, KvCacheSession destination) {
        requireOpen();
        long start = System.nanoTime();
        try {
            synchronized (prefixCache) {
                StoredPrefixEntry best = null;
                int limit = settings.getMaxPrefixTokensPerPrompt();
                for (int prefixLen : checkpointLengths(Math.min(tokens.length, limit))) {
                    StoredPrefixEntry entry = prefixCache.get(new CacheKey(salt, prefixTokens(tokens, prefixLen)));
                    if (entry != null) {
                        best = entry;
                    }
                }
                InferenceProfiler.counter(metricRegistry, "kvcache.v2.prefix.lookup").inc();
                if (best != null && best.length() >= blockSize && best.length() % blockSize == 0) {
                    best.restoreTo(destination);
                    InferenceProfiler.counter(metricRegistry, "kvcache.v2.prefix.hits").inc();
                    return new PrefixHit(best.length());
                }
                InferenceProfiler.counter(metricRegistry, "kvcache.v2.prefix.misses").inc();
                return null;
            }
        } finally {
            InferenceProfiler.timer(metricRegistry, "kvcache.v2.prefix.lookup.time")
                    .update(System.nanoTime() - start, TimeUnit.NANOSECONDS);
        }
    }

    public void storePrefix(int[] tokens, KvCacheSession source, Optional<String> salt) {
        requireOpen();
        if (settings.getMaxEntries() < 1) {
            return;
        }
        long start = System.nanoTime();
        try {
            int limit = settings.getMaxPrefixTokensPerPrompt();
            for (int prefixLen : checkpointLengths(Math.min(tokens.length, limit))) {
                if (prefixLen > source.length()) {
                    continue;
                }
                CacheKey key = new CacheKey(salt, prefixTokens(tokens, prefixLen));
                synchronized (prefixCache) {
                    if (prefixCache.containsKey(key)) {
                        continue;
                    }
                }
                StoredPrefixEntry entry = snapshot(source, prefixLen);
                boolean inserted = false;
                synchronized (prefixCache) {
                    if (!prefixCache.containsKey(key)) {
                        prefixCache.put(key, entry);
                        inserted = true;
                    }
                }
                if (!inserted) {
                    entry.close();
                }
            }
        } finally {
            InferenceProfiler.timer(metricRegistry, "kvcache.v2.prefix.store")
                    .update(System.nanoTime() - start, TimeUnit.NANOSECONDS);
        }
    }

    List<Integer> checkpointLengths(int tokenLength) {
        if (tokenLength < blockSize) {
            return List.of();
        }
        int largest = (tokenLength / blockSize) * blockSize;
        if (largest < blockSize) {
            return List.of();
        }
        if (settings.getPrefixCheckpointPolicy() == KvBufferCacheSettings.PrefixCheckpointPolicy.FIXED_BLOCKS) {
            ArrayList<Integer> fixed = new ArrayList<>();
            for (int prefixLen = blockSize; prefixLen <= largest; prefixLen += blockSize) {
                fixed.add(prefixLen);
            }
            return fixed;
        }
        if (settings.getPrefixCheckpointPolicy() == KvBufferCacheSettings.PrefixCheckpointPolicy.START_AND_END) {
            int max = settings.getMaxPrefixCheckpointsPerPrompt();
            int startCount = (max + 1) / 2;
            int endCount = max - startCount;
            LinkedHashSet<Integer> selected = new LinkedHashSet<>();
            for (int prefixLen = blockSize; prefixLen <= largest && selected.size() < startCount; prefixLen += blockSize) {
                selected.add(prefixLen);
            }
            for (int prefixLen = largest - ((endCount - 1) * blockSize); prefixLen <= largest; prefixLen += blockSize) {
                if (prefixLen >= blockSize) {
                    selected.add(prefixLen);
                }
            }
            return selected.stream().sorted().toList();
        }
        int max = settings.getMaxPrefixCheckpointsPerPrompt();
        LinkedHashSet<Integer> selected = new LinkedHashSet<>();
        for (Integer anchor : settings.getPrefixCheckpointAnchors()) {
            int aligned = (anchor / blockSize) * blockSize;
            if (aligned >= blockSize && aligned <= largest) {
                selected.add(aligned);
            }
            if (selected.size() >= Math.max(0, max - 1)) {
                break;
            }
        }
        selected.add(largest);
        ArrayList<Integer> result = new ArrayList<>(selected);
        result.sort(Integer::compareTo);
        if (result.size() > max) {
            ArrayList<Integer> trimmed = new ArrayList<>(result.subList(0, max - 1));
            trimmed.add(largest);
            return trimmed.stream().distinct().sorted().toList();
        }
        return result;
    }

    private StoredPrefixEntry snapshot(KvCacheSession source, int prefixLen) {
        long start = System.nanoTime();
        AbstractTensor[] keys = new AbstractTensor[layers];
        AbstractTensor[] values = new AbstractTensor[layers];
        try {
            for (int layer = 0; layer < layers; layer++) {
                keys[layer] = allocator.getDirty(keyDType, TensorShape.of(prefixLen, kvLength));
                values[layer] = allocator.getDirty(valueDType, TensorShape.of(prefixLen, kvLength));
                source.copyKeyRows(layer, 0, prefixLen, keys[layer], 0);
                source.copyValueRows(layer, 0, prefixLen, values[layer], 0);
            }
            long bytes = (long) layers * prefixLen * kvLength * keyDType.size()
                    + (long) layers * prefixLen * kvLength * valueDType.size();
            InferenceProfiler.counter(metricRegistry, "kvcache.v2.prefix.snapshot.bytes").inc(bytes);
            return new DenseStoredPrefixEntry(prefixLen, keys, values);
        } catch (RuntimeException | Error e) {
            for (AbstractTensor key : keys) {
                if (key != null) {
                    key.close();
                }
            }
            for (AbstractTensor value : values) {
                if (value != null) {
                    value.close();
                }
            }
            throw e;
        } finally {
            InferenceProfiler.timer(metricRegistry, "kvcache.v2.prefix.snapshot")
                    .update(System.nanoTime() - start, TimeUnit.NANOSECONDS);
        }
    }

    private static List<Integer> prefixTokens(int[] tokens, int prefixLen) {
        ArrayList<Integer> prefix = new ArrayList<>(prefixLen);
        for (int i = 0; i < prefixLen; i++) {
            prefix.add(tokens[i]);
        }
        return prefix;
    }

    private void requireOpen() {
        if (closed.get()) {
            throw new IllegalStateException("KVCache2 prefix snapshot cache is closed");
        }
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            synchronized (prefixCache) {
                prefixCache.values().forEach(StoredPrefixEntry::close);
                prefixCache.clear();
            }
        }
    }
}
