package io.teknek.deliverance.tensor.kv;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KvPrefixSnapshotCacheTest {
    private final MetricRegistry metricRegistry = new MetricRegistry();
    private final TensorAllocator allocator = new ArrayQueueTensorAllocator(metricRegistry);

    @Test
    void exactBlockHitRestoresRowsAndAdvancesLength() {
        KvBufferCacheSettings settings = settings();
        KvPrefixSnapshotCache cache = cache(settings);
        try (KvCacheSession source = session(settings);
             KvCacheSession restored = session(settings)) {
            fillSession(source, 8, 2);

            cache.storePrefix(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, source, Optional.empty());
            KvPrefixSnapshotCache.PrefixHit hit = cache.lookupPrefix(
                    new int[]{1, 2, 3, 4, 5, 6, 7, 8, 99}, Optional.empty(), restored);

            assertEquals(8, hit.length());
            assertEquals(8, restored.length());
            assertPrefixEquals(source, restored, 8, 2);
            assertEquals(1, metricRegistry.counter("kvcache.v2.prefix.hits").getCount());
        } finally {
            cache.close();
        }
    }

    @Test
    void longestMatchingPrefixWins() {
        KvBufferCacheSettings settings = settings();
        KvPrefixSnapshotCache cache = cache(settings);
        try (KvCacheSession source = session(settings);
             KvCacheSession restored = session(settings)) {
            fillSession(source, 12, 2);

            cache.storePrefix(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, source, Optional.empty());
            KvPrefixSnapshotCache.PrefixHit hit = cache.lookupPrefix(
                    new int[]{1, 2, 3, 4, 5, 6, 7, 8, 99}, Optional.empty(), restored);

            assertEquals(8, hit.length());
            assertEquals(8, restored.length());
        } finally {
            cache.close();
        }
    }

    @Test
    void noHitBelowBlockSize() {
        KvBufferCacheSettings settings = settings();
        KvPrefixSnapshotCache cache = cache(settings);
        try (KvCacheSession source = session(settings);
             KvCacheSession restored = session(settings)) {
            fillSession(source, 3, 2);

            cache.storePrefix(new int[]{1, 2, 3}, source, Optional.empty());

            assertNull(cache.lookupPrefix(new int[]{1, 2, 3}, Optional.empty(), restored));
            assertEquals(0, restored.length());
        } finally {
            cache.close();
        }
    }

    @Test
    void saltIsolatesEntries() {
        KvBufferCacheSettings settings = settings();
        KvPrefixSnapshotCache cache = cache(settings);
        try (KvCacheSession source = session(settings);
             KvCacheSession restored = session(settings)) {
            fillSession(source, 4, 2);
            int[] tokens = {1, 2, 3, 4};

            cache.storePrefix(tokens, source, Optional.of("a"));

            assertNull(cache.lookupPrefix(tokens, Optional.of("b"), restored));
            assertEquals(0, restored.length());
            assertEquals(4, cache.lookupPrefix(tokens, Optional.of("a"), restored).length());
        } finally {
            cache.close();
        }
    }

    @Test
    void evictionRemovesOldEntry() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withMaxEntries(1)
                .withBlockSize(4)
                .withPrefixCheckpointPolicy(KvBufferCacheSettings.PrefixCheckpointPolicy.FIXED_BLOCKS);
        KvPrefixSnapshotCache cache = cache(settings);
        try (KvCacheSession first = session(settings);
             KvCacheSession second = session(settings);
             KvCacheSession restored = session(settings)) {
            fillSession(first, 4, 2);
            fillSession(second, 4, 2);

            cache.storePrefix(new int[]{1, 2, 3, 4}, first, Optional.empty());
            cache.storePrefix(new int[]{5, 6, 7, 8}, second, Optional.empty());

            assertNull(cache.lookupPrefix(new int[]{1, 2, 3, 4}, Optional.empty(), restored));
            assertEquals(1, metricRegistry.counter("kvcache.v2.prefix.evict").getCount());
        } finally {
            cache.close();
        }
    }

    @Test
    void restoredPrefixCanAppendSuffix() {
        KvBufferCacheSettings settings = settings();
        KvPrefixSnapshotCache cache = cache(settings);
        try (KvCacheSession source = session(settings);
             KvCacheSession restored = session(settings)) {
            fillSession(source, 8, 2);
            cache.storePrefix(new int[]{1, 2, 3, 4, 5, 6, 7, 8}, source, Optional.empty());

            cache.lookupPrefix(new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9}, Optional.empty(), restored);
            writePosition(restored, 8, 200.0f, 2);
            restored.advanceLength(9);

            assertEquals(9, restored.length());
            try (AbstractTensor key = restored.keyRowCopy(0, 8)) {
                assertRow(key, 201.0f, 4);
            }
        } finally {
            cache.close();
        }
    }

    @Test
    void duplicateConcurrentStoresDoNotCorruptEntry() throws Exception {
        KvBufferCacheSettings settings = settings();
        KvPrefixSnapshotCache cache = cache(settings);
        try (KvCacheSession source = session(settings);
             KvCacheSession restored = session(settings)) {
            fillSession(source, 8, 2);
            CountDownLatch start = new CountDownLatch(1);
            ExecutorService executor = Executors.newFixedThreadPool(4);
            try {
                Future<?>[] futures = new Future<?>[4];
                for (int i = 0; i < futures.length; i++) {
                    futures[i] = executor.submit(() -> {
                        try {
                            start.await();
                            cache.storePrefix(new int[]{1, 2, 3, 4, 5, 6, 7, 8}, source, Optional.empty());
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new RuntimeException(e);
                        }
                    });
                }
                start.countDown();
                for (Future<?> future : futures) {
                    future.get();
                }
            } finally {
                executor.shutdownNow();
            }

            assertEquals(8, cache.lookupPrefix(new int[]{1, 2, 3, 4, 5, 6, 7, 8}, Optional.empty(), restored).length());
            assertPrefixEquals(source, restored, 8, 2);
        } finally {
            cache.close();
        }
    }

    @Test
    void checkpointPoliciesMatchExistingPrefixCacheBehavior() {
        KvPrefixSnapshotCache fixed = cache(settings());
        try {
            assertEquals(List.of(), fixed.checkpointLengths(3));
            assertEquals(List.of(4), fixed.checkpointLengths(4));
            assertEquals(List.of(4, 8, 12), fixed.checkpointLengths(15));
        } finally {
            fixed.close();
        }

        KvBufferCacheSettings anchoredSettings = new KvBufferCacheSettings(true)
                .withBlockSize(4)
                .withMaxPrefixTokensPerPrompt(1000)
                .withMaxPrefixCheckpointsPerPrompt(4)
                .withPrefixCheckpointPolicy(KvBufferCacheSettings.PrefixCheckpointPolicy.ANCHORS_AND_LARGEST)
                .withPrefixCheckpointAnchors(List.of(4, 8, 12));
        KvPrefixSnapshotCache anchored = cache(anchoredSettings);
        try {
            assertEquals(List.of(4, 8, 12, 32), anchored.checkpointLengths(35));
        } finally {
            anchored.close();
        }
    }

    private KvBufferCacheSettings settings() {
        return new KvBufferCacheSettings(true)
                .withMaxEntries(512)
                .withBlockSize(4)
                .withPrefixCheckpointPolicy(KvBufferCacheSettings.PrefixCheckpointPolicy.FIXED_BLOCKS);
    }

    private KvPrefixSnapshotCache cache(KvBufferCacheSettings settings) {
        return new KvPrefixSnapshotCache(2, 16, 4, settings.getBlockSize(), DType.F32, allocator, metricRegistry,
                settings);
    }

    private KvCacheSession session(KvBufferCacheSettings settings) {
        return new KvCacheManager(2, 16, 4, DType.F32, settings, allocator, metricRegistry).openSession();
    }

    private void fillSession(KvCacheSession session, int length, int layers) {
        for (int position = 0; position < length; position++) {
            writePosition(session, position, 10.0f * position, layers);
            session.advanceLength(position + 1);
        }
    }

    private void writePosition(KvCacheSession session, int position, float base, int layers) {
        try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
            for (int layer = 0; layer < layers; layer++) {
                try (AbstractTensor key = row(base + layer * 100.0f + 1.0f);
                     AbstractTensor value = row(base + layer * 100.0f + 2.0f)) {
                    writer.write(layer, position, key, value);
                }
            }
        }
    }

    private AbstractTensor row(float firstValue) {
        AbstractTensor tensor = allocator.getDirty(DType.F32, TensorShape.of(1, 4));
        for (int i = 0; i < 4; i++) {
            tensor.set(firstValue + i, 0, i);
        }
        return tensor;
    }

    private void assertPrefixEquals(KvCacheSession expected, KvCacheSession actual, int length, int layers) {
        for (int layer = 0; layer < layers; layer++) {
            for (int position = 0; position < length; position++) {
                try (AbstractTensor expectedKey = expected.keyRowCopy(layer, position);
                     AbstractTensor actualKey = actual.keyRowCopy(layer, position);
                     AbstractTensor expectedValue = expected.valueRowCopy(layer, position);
                     AbstractTensor actualValue = actual.valueRowCopy(layer, position)) {
                    for (int i = 0; i < 4; i++) {
                        assertEquals(expectedKey.get(0, i), actualKey.get(0, i), 0.0f);
                        assertEquals(expectedValue.get(0, i), actualValue.get(0, i), 0.0f);
                    }
                }
            }
        }
    }

    private static void assertRow(AbstractTensor tensor, float firstValue, int width) {
        for (int i = 0; i < width; i++) {
            assertEquals(firstValue + i, tensor.get(0, i), 0.0f, "col=" + i);
        }
    }
}
