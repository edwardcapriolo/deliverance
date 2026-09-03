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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KvBlockDiskStoreTurboQuantIT {
    private final MetricRegistry metricRegistry = new MetricRegistry();
    private final TensorAllocator allocator = new ArrayQueueTensorAllocator(metricRegistry);

    @TempDir
    Path tempDir;

    @Test
    void smallPromptTurboQuantBlockPersistsExactlyAsTurboQuantInTempDirectory() throws Exception {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withPrefixCacheMode(KvBufferCacheSettings.PrefixCacheMode.SHARED_BLOCKS)
                .withBlockSize(2)
                .withKvBlockStoragePolicy(KvBufferCacheSettings.KvBlockStoragePolicy.MSE_TURBOQUANT)
                .withKvTurboQuantBits(4)
                .withSharedPrefixDiskCacheEnabled(true)
                .withSharedPrefixDiskCachePath(tempDir.toFile())
                .withSharedPrefixDiskCacheMaxBytes(KvBlockDiskStore.MIN_DISK_CACHE_BYTES)
                .withSharedPrefixDiskCacheMinUsableBytes(0)
                .withSharedPrefixDiskCacheReservedFreeBytes(0)
                .withSharedPrefixDiskCacheAdmitMinTokens(2)
                .withSharedPrefixDiskCacheWriterQueueSize(8);
        KvBlockKey key = key();

        KvBlockManager writerManager = new KvBlockManager(metricRegistry, settings, allocator);
        try (KvCacheSession session = session(settings)) {
            writeSmallPrompt(session);
            KvBlock block = session.detachCommittedBlock(0);
            assertEquals(KvBlockLayout.MSE_TURBOQUANT, block.layout());
            session.attachCommittedBlock(writerManager.admitAndRetain(key, block, session.sessionId(), 2));
        }
        writerManager.close();

        assertEquals(1, countNamespaceDirectories());
        assertEquals(1, countFilesWithSuffix(".bin"));
        assertEquals(1, countFilesWithSuffix(".meta.json"));

        KvBlockManager readerManager = new KvBlockManager(metricRegistry, settings, allocator);
        try (KvCacheSession session = session(settings)) {
            KvBlockLease lease = readerManager.retain(key, session.sessionId());
            assertNotNull(lease);
            assertEquals(KvBlockLayout.MSE_TURBOQUANT, lease.block().layout());
            session.attachCommittedBlock(lease);
            assertEquals(2, session.length());
            try (AbstractTensor keyRow = session.keyRowCopy(0, 1);
                 AbstractTensor valueRow = session.valueRowCopy(0, 1)) {
                assertFalse(Float.isNaN(keyRow.get(0, 0)));
                assertFalse(Float.isNaN(valueRow.get(0, 0)));
                assertTrue(Math.abs(keyRow.get(0, 0)) > 0.01f);
                assertTrue(Math.abs(valueRow.get(0, 0)) > 0.01f);
            }
        }
        readerManager.close();
    }

    private KvCacheSession session(KvBufferCacheSettings settings) {
        return new KvCacheSession(1, 8, 32, 2, DType.F32, allocator, metricRegistry, false, settings, null);
    }

    private void writeSmallPrompt(KvCacheSession session) {
        try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
            for (int position = 0; position < 2; position++) {
                try (AbstractTensor key = row(position, 0);
                     AbstractTensor value = row(position, 1)) {
                    writer.write(0, position, key, value);
                }
            }
            writer.advanceLength(2);
        }
    }

    private AbstractTensor row(int position, int keyOrValue) {
        AbstractTensor tensor = allocator.getDirty(DType.F32, TensorShape.of(1, 32));
        for (int i = 0; i < 32; i++) {
            float value = (float) (Math.sin((position + 1) * (i + 1) * 0.17)
                    + Math.cos((keyOrValue + 1) * (i + 3) * 0.11)
                    + position * 0.25
                    + keyOrValue * 0.5);
            tensor.set(value, 0, i);
        }
        return tensor;
    }

    private KvBlockKey key() {
        return new KvBlockKey(1, "test-model", "none", "test-tokenizer", "", "test-rope", "test-attention",
                0, 0L, 10L, 2, 2, 1, 32, DType.F32, DType.F32, KvBlockLayout.MSE_TURBOQUANT, 4,
                1, 0, 0L, "local");
    }

    private long countFilesWithSuffix(String suffix) throws IOException {
        try (var paths = Files.walk(tempDir)) {
            return paths.filter(path -> Files.isRegularFile(path) && path.getFileName().toString().endsWith(suffix))
                    .count();
        }
    }

    private long countNamespaceDirectories() throws IOException {
        try (var paths = Files.list(tempDir)) {
            return paths.filter(Files::isDirectory).count();
        }
    }
}
