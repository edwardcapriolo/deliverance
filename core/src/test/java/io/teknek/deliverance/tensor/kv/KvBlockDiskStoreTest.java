package io.teknek.deliverance.tensor.kv;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

class KvBlockDiskStoreTest {
    private final MetricRegistry metricRegistry = new MetricRegistry();
    private final TensorAllocator allocator = new ArrayQueueTensorAllocator(metricRegistry);

    @TempDir
    Path tempDir;

    @Test
    void denseI8BlockPersistsAndReloadsThroughFreshManager() {
        KvBufferCacheSettings settings = diskSettings();
        KvBlockKey key = key(0, 0L, 10L);

        KvBlockManager writerManager = new KvBlockManager(metricRegistry, settings, allocator);
        try (KvCacheSession session = session(settings)) {
            writeFullBlock(session, 10.0f);
            KvBlock block = session.detachCommittedBlock(0);
            session.attachCommittedBlock(writerManager.admitAndRetain(key, block, session.sessionId()));
        }
        writerManager.close();

        KvBlockManager readerManager = new KvBlockManager(metricRegistry, settings, allocator);
        try (KvCacheSession session = session(settings)) {
            KvBlockLease lease = readerManager.retain(key, session.sessionId());
            assertNotNull(lease);
            session.attachCommittedBlock(lease);
            assertEquals(2, session.length());
            try (AbstractTensor keyRow = session.keyRowCopy(0, 1);
                 AbstractTensor valueRow = session.valueRowCopy(0, 1)) {
                assertEquals(DType.I8, keyRow.dType());
                assertEquals(DType.I8, valueRow.dType());
                assertEquals(21.0f, keyRow.get(0, 0), 0.5f);
                assertEquals(22.0f, valueRow.get(0, 0), 0.5f);
            }
        }
        readerManager.close();
    }

    @Test
    void wrongKeyMissesDiskCache() {
        KvBufferCacheSettings settings = diskSettings();
        KvBlockKey key = key(0, 0L, 10L);
        KvBlockKey wrongKey = key(0, 0L, 11L);

        KvBlockManager writerManager = new KvBlockManager(metricRegistry, settings, allocator);
        try (KvCacheSession session = session(settings)) {
            writeFullBlock(session, 10.0f);
            session.attachCommittedBlock(writerManager.admitAndRetain(key, session.detachCommittedBlock(0),
                    session.sessionId()));
        }
        writerManager.close();

        KvBlockManager readerManager = new KvBlockManager(metricRegistry, settings, allocator);
        try (KvCacheSession session = session(settings)) {
            assertNull(readerManager.retain(wrongKey, session.sessionId()));
        }
        readerManager.close();
    }

    @Test
    void corruptedPayloadMissesDiskCache() throws Exception {
        KvBufferCacheSettings settings = diskSettings();
        KvBlockKey key = key(0, 0L, 10L);

        KvBlockManager writerManager = new KvBlockManager(metricRegistry, settings, allocator);
        try (KvCacheSession session = session(settings)) {
            writeFullBlock(session, 10.0f);
            session.attachCommittedBlock(writerManager.admitAndRetain(key, session.detachCommittedBlock(0),
                    session.sessionId()));
        }
        writerManager.close();
        Path bin = firstFileWithSuffix(".bin");
        Files.write(bin, new byte[] {1, 2, 3});

        KvBlockManager readerManager = new KvBlockManager(metricRegistry, settings, allocator);
        try (KvCacheSession session = session(settings)) {
            assertNull(readerManager.retain(key, session.sessionId()));
        }
        readerManager.close();
    }

    @Test
    void diskStoreDisablesWhenMaxBytesIsTooSmall() {
        KvBufferCacheSettings settings = diskSettings().withSharedPrefixDiskCacheMaxBytes(1024);

        assertNull(KvBlockDiskStore.open(settings, allocator, metricRegistry));
    }

    @Test
    void diskAdmitThresholdUsesPromptPrefixTokensNotIndividualBlockSize() throws Exception {
        KvBufferCacheSettings settings = diskSettings().withSharedPrefixDiskCacheAdmitMinTokens(128);
        KvBlockKey key = key(0, 0L, 10L);

        KvBlockManager writerManager = new KvBlockManager(metricRegistry, settings, allocator);
        try (KvCacheSession session = session(settings)) {
            writeFullBlock(session, 10.0f);
            KvBlock block = session.detachCommittedBlock(0);
            session.attachCommittedBlock(writerManager.admitAndRetain(key, block, session.sessionId(), 128));
        }
        writerManager.close();

        long binCount = countFilesWithSuffix(".bin");
        assertEquals(1, binCount);
    }

    @Test
    void modelFingerprintsAreStoredInSeparateNamespaceDirectories() throws Exception {
        KvBufferCacheSettings settings = diskSettings();
        KvBlockKey firstModelKey = key("test-model-a", 0, 0L, 10L);
        KvBlockKey secondModelKey = key("test-model-b", 0, 0L, 10L);

        KvBlockManager writerManager = new KvBlockManager(metricRegistry, settings, allocator);
        try (KvCacheSession session = session(settings)) {
            writeFullBlock(session, 10.0f);
            session.attachCommittedBlock(writerManager.admitAndRetain(firstModelKey, session.detachCommittedBlock(0),
                    session.sessionId()));
        }
        try (KvCacheSession session = session(settings)) {
            writeFullBlock(session, 20.0f);
            session.attachCommittedBlock(writerManager.admitAndRetain(secondModelKey, session.detachCommittedBlock(0),
                    session.sessionId()));
        }
        writerManager.close();

        assertEquals(2, namespaceDirectories().size());
        assertEquals(2, countFilesWithSuffix(".bin"));
        assertEquals(2, countFilesWithSuffix(".meta.json"));
    }

    private KvBufferCacheSettings diskSettings() {
        return new KvBufferCacheSettings(true)
                .withPrefixCacheMode(KvBufferCacheSettings.PrefixCacheMode.SHARED_BLOCKS)
                .withBlockSize(2)
                .withKvKeyDType(DType.I8)
                .withKvValueDType(DType.I8)
                .withSharedPrefixDiskCacheEnabled(true)
                .withSharedPrefixDiskCachePath(tempDir.toFile())
                .withSharedPrefixDiskCacheMaxBytes(KvBlockDiskStore.MIN_DISK_CACHE_BYTES)
                .withSharedPrefixDiskCacheMinUsableBytes(0)
                .withSharedPrefixDiskCacheReservedFreeBytes(0)
                .withSharedPrefixDiskCacheAdmitMinTokens(0)
                .withSharedPrefixDiskCacheWriterQueueSize(8);
    }

    private KvCacheSession session(KvBufferCacheSettings settings) {
        return new KvCacheSession(1, 8, 32, 2, DType.F32, allocator, metricRegistry, false, settings, null);
    }

    private void writeFullBlock(KvCacheSession session, float base) {
        try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
            for (int position = 0; position < 2; position++) {
                try (AbstractTensor key = row(base + position * 10.0f + 1.0f);
                     AbstractTensor value = row(base + position * 10.0f + 2.0f)) {
                    writer.write(0, position, key, value);
                }
            }
            writer.advanceLength(2);
        }
    }

    private AbstractTensor row(float firstValue) {
        AbstractTensor tensor = allocator.getDirty(DType.F32, TensorShape.of(1, 32));
        for (int i = 0; i < 32; i++) {
            tensor.set(firstValue + i, 0, i);
        }
        return tensor;
    }

    private KvBlockKey key(int blockIndex, long parentHash, long blockHash) {
        return key("test-model", blockIndex, parentHash, blockHash);
    }

    private KvBlockKey key(String modelCacheId, int blockIndex, long parentHash, long blockHash) {
        return new KvBlockKey(1, modelCacheId, "none", "test-tokenizer", "", "test-rope", "test-attention",
                blockIndex, parentHash, blockHash, 2, 2, 1, 32, DType.I8, DType.I8, KvBlockLayout.DENSE, 0,
                1, 0, 0L, "local");
    }

    private Path firstFileWithSuffix(String suffix) throws IOException {
        try (var paths = Files.walk(tempDir)) {
            return paths.filter(path -> Files.isRegularFile(path) && path.getFileName().toString().endsWith(suffix))
                    .findFirst().orElseThrow();
        }
    }

    private long countFilesWithSuffix(String suffix) throws IOException {
        try (var paths = Files.walk(tempDir)) {
            return paths.filter(path -> Files.isRegularFile(path) && path.getFileName().toString().endsWith(suffix))
                    .count();
        }
    }

    private Set<Path> namespaceDirectories() throws IOException {
        try (var paths = Files.list(tempDir)) {
            return paths.filter(Files::isDirectory).collect(Collectors.toSet());
        }
    }
}
