package io.teknek.deliverance.tensor.kv;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.ReadOnlyTensor;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.TrackedReadOnlyTensor;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class KvCacheSessionTest {
    private final MetricRegistry metricRegistry = new MetricRegistry();
    private final TensorAllocator allocator = new ArrayQueueTensorAllocator(metricRegistry);

    @Test
    void writeReadAndCommitImmutableBlocks() {
        KvCacheManager manager = new KvCacheManager(2, 8, 4, DType.F32,
                new KvBufferCacheSettings(true).withBlockSize(2), allocator, metricRegistry);

        try (KvCacheSession session = manager.openSession()) {
            writePosition(session, 0, 10.0f);
            writePosition(session, 1, 20.0f);
            try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
                writer.advanceLength(2);
            }

            assertEquals(2, session.length());
            assertEquals(1, session.committedBlocks().size());
            assertEquals(0, session.committedBlocks().getFirst().blockIndex());

            try (AbstractTensor key = session.keyRowCopy(1, 1);
                 AbstractTensor value = session.valueRowCopy(1, 1)) {
                assertRow(key, 121.0f);
                assertRow(value, 122.0f);
            }

            try (AbstractTensor key = row(999.0f);
                 AbstractTensor value = row(1000.0f);
                 KvWriteCursor writer = session.writer(CacheExecutionMode.DECODE_UPDATE_CACHE)) {
                assertThrows(IllegalArgumentException.class, () -> writer.write(0, 0, key, value));
            }
        }
    }

    @Test
    void readViewCopiesVisibleRowsInLogicalOrder() {
        KvCacheManager manager = new KvCacheManager(1, 8, 4, DType.F32,
                new KvBufferCacheSettings(true).withBlockSize(2), allocator, metricRegistry);

        try (KvCacheSession session = manager.openSession()) {
            writePosition(session, 0, 1.0f, 1);
            writePosition(session, 1, 2.0f, 1);
            writePosition(session, 2, 3.0f, 1);
            try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
                writer.advanceLength(3);
            }

            try (KvReadView view = session.readView(0, 3, AttentionPattern.CAUSAL);
                 AbstractTensor keys = view.copyVisibleKeys();
                 AbstractTensor values = view.copyVisibleValues()) {
                assertEquals(0, view.layer());
                assertEquals(3, view.visibleTokens());
                assertEquals(AttentionPattern.CAUSAL, view.pattern());
                assertEquals(2.0f, keys.get(0, 0), 0.0f);
                assertEquals(3.0f, keys.get(1, 0), 0.0f);
                assertEquals(4.0f, keys.get(2, 0), 0.0f);
                assertEquals(3.0f, values.get(0, 0), 0.0f);
                assertEquals(4.0f, values.get(1, 0), 0.0f);
                assertEquals(5.0f, values.get(2, 0), 0.0f);
            }
        }
    }

    @Test
    void readViewRowReturnsNonCopyingReadOnlyView() {
        KvCacheManager manager = new KvCacheManager(1, 8, 4, DType.F32,
                new KvBufferCacheSettings(true).withBlockSize(1), allocator, metricRegistry);

        try (KvCacheSession session = manager.openSession()) {
            writePosition(session, 0, 10.0f, 1);
            try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
                writer.advanceLength(1);
            }

            try (KvReadView readView = session.readView(0, 1, AttentionPattern.CAUSAL);
                 AbstractTensor key = readView.keyRow(0)) {
                assertTrue(key instanceof ReadOnlyTensor);
                assertThrows(UnsupportedOperationException.class, () -> key.set(99.0f, 0, 0));
                key.getMemorySegment().set(ValueLayout.JAVA_FLOAT, key.getMemorySegmentOffset(0), 123.0f);
            }

            try (AbstractTensor keyCopy = session.keyRowCopy(0, 0)) {
                assertEquals(123.0f, keyCopy.get(0, 0), 0.0f);
            }
        }
    }

    @Test
    void trackedReadViewRowDetectsBackingMutationOnClose() {
        KvCacheManager manager = new KvCacheManager(1, 8, 4, DType.F32,
                new KvBufferCacheSettings(true).withBlockSize(1), allocator, metricRegistry, true);

        try (KvCacheSession session = manager.openSession()) {
            writePosition(session, 0, 10.0f, 1);
            try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
                writer.advanceLength(1);
            }

            try (KvReadView readView = session.readView(0, 1, AttentionPattern.CAUSAL)) {
                AbstractTensor key = readView.keyRow(0);
                assertTrue(key instanceof TrackedReadOnlyTensor);
                TrackedReadOnlyTensor tracked = (TrackedReadOnlyTensor) key;
                assertFalse(tracked.hasChecksumChanged());

                key.getMemorySegment().set(ValueLayout.JAVA_FLOAT, key.getMemorySegmentOffset(0), 123.0f);

                assertTrue(tracked.hasChecksumChanged());
                assertThrows(IllegalStateException.class, tracked::close);
            }
        }
    }

    @Test
    void turboQuantCommittedBlockUsesCompressedLayoutAndDecodedRowsStayWithinDistribution() {
        int layers = 2;
        int blockSize = 4;
        int kvLength = 64;
        KvCacheManager manager = new KvCacheManager(layers, 8, kvLength, DType.F32,
                new KvBufferCacheSettings(true)
                        .withBlockSize(blockSize)
                        .withKvBlockStoragePolicy(KvBufferCacheSettings.KvBlockStoragePolicy.MSE_TURBOQUANT)
                        .withKvTurboQuantBits(4),
                allocator, metricRegistry, true);
        float[][][][] expected = new float[layers][2][blockSize][kvLength];

        try (KvCacheSession session = manager.openSession()) {
            try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
                for (int position = 0; position < blockSize; position++) {
                    for (int layer = 0; layer < layers; layer++) {
                        try (AbstractTensor key = statisticalRow(layer, position, 0, kvLength, expected);
                             AbstractTensor value = statisticalRow(layer, position, 1, kvLength, expected)) {
                            writer.write(layer, position, key, value);
                        }
                    }
                }
                writer.advanceLength(blockSize);
            }

            KvBlock block = session.committedBlocks().getFirst();
            assertEquals(KvBlockLayout.MSE_TURBOQUANT, block.layout());
            assertTrue(block.encodedBytes() < block.denseBytesEquivalent() / 2,
                    "expected TurboQuant KV block to be materially smaller");

            double squaredError = 0.0;
            double sum = 0.0;
            double sumSquares = 0.0;
            int count = 0;
            for (int layer = 0; layer < layers; layer++) {
                try (KvReadView readView = session.readView(layer, blockSize, AttentionPattern.CAUSAL)) {
                    for (int position = 0; position < blockSize; position++) {
                        try (AbstractTensor key = readView.keyRow(position);
                             AbstractTensor value = readView.valueRow(position)) {
                            for (int i = 0; i < kvLength; i++) {
                                double expectedKey = expected[layer][0][position][i];
                                double expectedValue = expected[layer][1][position][i];
                                squaredError += square(expectedKey - key.get(0, i));
                                squaredError += square(expectedValue - value.get(0, i));
                                sum += expectedKey + expectedValue;
                                sumSquares += expectedKey * expectedKey + expectedValue * expectedValue;
                                count += 2;
                            }
                        }
                    }
                }
            }
            double rmse = Math.sqrt(squaredError / count);
            double mean = sum / count;
            double standardDeviation = Math.sqrt((sumSquares / count) - (mean * mean));
            assertTrue(rmse < standardDeviation,
                    "TurboQuant reconstruction RMSE should remain within one standard deviation: rmse="
                            + rmse + " stddev=" + standardDeviation);
        }
    }

    @Test
    void denseKvCanStoreKeysAndValuesAsI8() {
        int kvLength = 64;
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withBlockSize(4)
                .withKvKeyDType(DType.I8)
                .withKvValueDType(DType.I8);
        KvCacheManager manager = new KvCacheManager(1, 8, kvLength, DType.F32, settings, allocator, metricRegistry);

        try (KvCacheSession session = manager.openSession()) {
            try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
                for (int position = 0; position < 4; position++) {
                    try (AbstractTensor key = wideRow(position + 1.0f, kvLength);
                         AbstractTensor value = wideRow(position + 2.0f, kvLength)) {
                        writer.write(0, position, key, value);
                    }
                }
                writer.advanceLength(4);
            }

            try (KvReadView readView = session.readView(0, 4, AttentionPattern.CAUSAL);
                 AbstractTensor key = readView.keyRow(0);
                 AbstractTensor value = readView.valueRow(0)) {
                assertEquals(DType.I8, key.dType());
                assertEquals(DType.I8, value.dType());
            }

            assertEquals(KvBlockLayout.DENSE, session.committedBlocks().getFirst().layout());
            assertTrue(session.committedBlocks().getFirst().encodedBytes() < 4L * 2 * kvLength * DType.F32.size());
        }
    }

    @Test
    void denseKvCanStoreKeysAndValuesAsBf16() {
        int kvLength = 64;
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withBlockSize(4)
                .withKvKeyDType(DType.BF16)
                .withKvValueDType(DType.BF16);
        KvCacheManager manager = new KvCacheManager(1, 8, kvLength, DType.F32, settings, allocator, metricRegistry);

        try (KvCacheSession session = manager.openSession()) {
            try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
                for (int position = 0; position < 4; position++) {
                    try (AbstractTensor key = wideRow(position + 1.0f, kvLength);
                         AbstractTensor value = wideRow(position + 2.0f, kvLength)) {
                        writer.write(0, position, key, value);
                    }
                }
                writer.advanceLength(4);
            }

            try (KvReadView readView = session.readView(0, 4, AttentionPattern.CAUSAL);
                 AbstractTensor key = readView.keyRow(0);
                 AbstractTensor value = readView.valueRow(0)) {
                assertEquals(DType.BF16, key.dType());
                assertEquals(DType.BF16, value.dType());
                assertEquals(1.0f, key.get(0, 0), 0.01f);
                assertEquals(2.0f, value.get(0, 0), 0.01f);
            }

            assertEquals(KvBlockLayout.DENSE, session.committedBlocks().getFirst().layout());
            assertEquals(4L * 2 * kvLength * DType.BF16.size(),
                    session.committedBlocks().getFirst().encodedBytes());
        }
    }

    @Test
    void denseKvConversionUsesSuppliedTensorOperations() {
        int kvLength = 64;
        TensorOperations operations = Mockito.mock(TensorOperations.class);
        when(operations.quantize(any(AbstractTensor.class), eq(DType.I8), eq(0), eq(kvLength)))
                .thenAnswer(invocation -> AbstractTensorUtils.quantize(invocation.getArgument(0), DType.I8, true));
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withBlockSize(4)
                .withKvKeyDType(DType.I8)
                .withKvValueDType(DType.I8);
        KvCacheManager manager = new KvCacheManager(1, 8, kvLength, DType.F32, settings, allocator,
                metricRegistry, false, operations);

        try (KvCacheSession session = manager.openSession();
             KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE);
             AbstractTensor key = wideRow(1.0f, kvLength);
             AbstractTensor value = wideRow(2.0f, kvLength)) {
            writer.write(0, 0, key, value);
        }

        verify(operations, times(2)).quantize(any(AbstractTensor.class), eq(DType.I8), eq(0), eq(kvLength));
    }

    @Test
    void i8KvIsRejectedWithTurboQuantPolicy() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true).withKvKeyDType(DType.I8);
        assertThrows(IllegalArgumentException.class,
                () -> settings.setKvBlockStoragePolicy(KvBufferCacheSettings.KvBlockStoragePolicy.MSE_TURBOQUANT));
    }

    @Test
    void denoiseModeCannotWrite() {
        KvCacheManager manager = new KvCacheManager(1, 8, 4, DType.F32,
                new KvBufferCacheSettings(true).withBlockSize(2), allocator, metricRegistry);

        try (KvCacheSession session = manager.openSession();
             AbstractTensor key = row(1.0f);
             AbstractTensor value = row(2.0f);
             KvWriteCursor writer = session.writer(CacheExecutionMode.DENOISE_BLOCK_NO_UPDATE)) {
            assertThrows(IllegalArgumentException.class, () -> writer.write(0, 0, key, value));
        }
    }

    @Test
    void cropRemovesMutableTailWithoutMutatingCommittedPrefix() {
        KvCacheManager manager = new KvCacheManager(1, 8, 4, DType.F32,
                new KvBufferCacheSettings(true).withBlockSize(2), allocator, metricRegistry);

        try (KvCacheSession session = manager.openSession()) {
            writePosition(session, 0, 1.0f, 1);
            writePosition(session, 1, 2.0f, 1);
            writePosition(session, 2, 3.0f, 1);
            try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
                writer.advanceLength(3);
            }

            session.crop(2);

            assertEquals(2, session.length());
            assertEquals(1, session.committedBlocks().size());
            try (AbstractTensor key = session.keyRowCopy(0, 1)) {
                assertRow(key, 3.0f);
            }
            assertThrows(IllegalArgumentException.class, () -> session.keyRowCopy(0, 2));
        }
    }

    @Test
    void cropCanSplitCommittedBlockAndAppendAfterward() {
        KvCacheManager manager = new KvCacheManager(1, 8, 4, DType.F32,
                new KvBufferCacheSettings(true).withBlockSize(4), allocator, metricRegistry);

        try (KvCacheSession session = manager.openSession()) {
            writePosition(session, 0, 10.0f, 1);
            writePosition(session, 1, 20.0f, 1);
            writePosition(session, 2, 30.0f, 1);
            writePosition(session, 3, 40.0f, 1);
            try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
                writer.advanceLength(4);
            }
            assertEquals(1, session.committedBlocks().size());

            session.crop(2);

            assertEquals(2, session.length());
            assertEquals(0, session.committedBlocks().size());
            try (AbstractTensor key = session.keyRowCopy(0, 1)) {
                assertRow(key, 21.0f);
            }
            assertThrows(IllegalArgumentException.class, () -> session.keyRowCopy(0, 2));

            writePosition(session, 2, 50.0f, 1);
            try (KvWriteCursor writer = session.writer(CacheExecutionMode.DECODE_UPDATE_CACHE)) {
                writer.advanceLength(3);
            }

            try (AbstractTensor key = session.keyRowCopy(0, 0);
                 AbstractTensor appended = session.keyRowCopy(0, 2)) {
                assertRow(key, 11.0f);
                assertRow(appended, 51.0f);
            }
        }
    }

    @Test
    void samePromptCanAttachLeasesToSameImmutableBlocks() {
        KvBlockManager blockManager = new KvBlockManager(metricRegistry);
        KvCacheManager sessionManager = new KvCacheManager(1, 8, 4, DType.F32,
                new KvBufferCacheSettings(true).withBlockSize(2), allocator, metricRegistry);
        List<KvBlockKey> keys;
        List<Integer> blockIdentities;

        try (KvCacheSession first = sessionManager.openSession()) {
            writeFourTokenPrompt(first);
            keys = blockKeys(first, new long[] {10L, 20L});
            blockIdentities = transferCommittedBlocksToManager(first, blockManager, keys);
            assertEquals(List.of(0, 1), first.attachedLeaseBlockIndexes());
        }

        assertEquals(2, blockManager.residentBlockCount());
        assertEquals(0, blockManager.refCount(keys.get(0)));
        assertEquals(0, blockManager.refCount(keys.get(1)));

        try (KvCacheSession second = sessionManager.openSession()) {
            for (KvBlockKey key : keys) {
                KvBlockLease lease = blockManager.retain(key, second.sessionId());
                assertNotNull(lease);
                second.attachCommittedBlock(lease);
            }

            assertEquals(4, second.length());
            assertEquals(List.of(0, 1), second.attachedLeaseBlockIndexes());
            assertEquals(blockIdentities.get(0), second.committedBlockLeases().get(0).blockIdentity());
            assertEquals(blockIdentities.get(1), second.committedBlockLeases().get(1).blockIdentity());
            assertEquals(1, blockManager.refCount(keys.get(0)));
            assertEquals(1, blockManager.refCount(keys.get(1)));
            try (AbstractTensor key = second.keyRowCopy(0, 3);
                 AbstractTensor value = second.valueRowCopy(0, 3)) {
                assertRow(key, 41.0f);
                assertRow(value, 42.0f);
            }
        }

        assertEquals(0, blockManager.refCount(keys.get(0)));
        assertEquals(0, blockManager.refCount(keys.get(1)));
        blockManager.close();
    }

    @Test
    void differentSuffixReusesOnlyCommonFullBlocks() {
        KvBlockManager blockManager = new KvBlockManager(metricRegistry);
        KvCacheManager sessionManager = new KvCacheManager(1, 8, 4, DType.F32,
                new KvBufferCacheSettings(true).withBlockSize(2), allocator, metricRegistry);

        List<KvBlockKey> firstKeys;
        try (KvCacheSession first = sessionManager.openSession()) {
            writeFourTokenPrompt(first);
            firstKeys = blockKeys(first, new long[] {10L, 20L});
            transferCommittedBlocksToManager(first, blockManager, firstKeys);
        }

        KvBlockKey commonPrefix = firstKeys.get(0);
        KvBlockKey differentSuffix = localKey(1, 10L, 99L, 2, 1, 4, KvBlockLayout.DENSE);
        try (KvCacheSession second = sessionManager.openSession()) {
            KvBlockLease commonLease = blockManager.retain(commonPrefix, second.sessionId());
            assertNotNull(commonLease);
            second.attachCommittedBlock(commonLease);
            assertNull(blockManager.retain(differentSuffix, second.sessionId()));
            assertEquals(2, second.length());
        }
        blockManager.close();
    }

    @Test
    void partialTailBlockIsNotShared() {
        KvBlockManager blockManager = new KvBlockManager(metricRegistry);
        KvCacheManager sessionManager = new KvCacheManager(1, 8, 4, DType.F32,
                new KvBufferCacheSettings(true).withBlockSize(4), allocator, metricRegistry);

        try (KvCacheSession session = sessionManager.openSession()) {
            for (int position = 0; position < 6; position++) {
                writePosition(session, position, (position + 1) * 10.0f, 1);
            }
            try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
                writer.advanceLength(6);
            }

            assertEquals(1, session.committedBlocks().size());
            List<KvBlockKey> keys = blockKeys(session, new long[] {10L});
            transferCommittedBlocksToManager(session, blockManager, keys);
            assertEquals(1, blockManager.residentBlockCount());
            assertNull(blockManager.retain(localKey(1, 10L, 20L, 4, 1, 4, KvBlockLayout.DENSE), session.sessionId()));
        }
        blockManager.close();
    }

    @Test
    void rankAwareIdentityPreventsCrossRankReuse() {
        KvBlockManager blockManager = new KvBlockManager(metricRegistry);
        KvCacheManager sessionManager = new KvCacheManager(1, 4, 4, DType.F32,
                new KvBufferCacheSettings(true).withBlockSize(2), allocator, metricRegistry);
        KvBlockKey rank0 = tpKey(0, 2, 0, 100L, 1, 0L, 10L, 2, 1, 4, KvBlockLayout.DENSE);
        KvBlockKey rank1 = tpKey(0, 2, 1, 100L, 1, 0L, 10L, 2, 1, 4, KvBlockLayout.DENSE);

        try (KvCacheSession session = sessionManager.openSession()) {
            writePosition(session, 0, 10.0f, 1);
            writePosition(session, 1, 20.0f, 1);
            try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
                writer.advanceLength(2);
            }
            KvBlock block = session.detachCommittedBlock(0);
            session.attachCommittedBlock(blockManager.admitAndRetain(rank0, block, session.sessionId()));

            KvBlockLease rank0Lease = blockManager.retain(rank0, session.sessionId());
            assertNotNull(rank0Lease);
            rank0Lease.close();
            assertNull(blockManager.retain(rank1, session.sessionId()));
        }
        blockManager.close();
    }

    @Test
    void assignmentEpochPreventsStaleRankReuse() {
        KvBlockManager blockManager = new KvBlockManager(metricRegistry);
        KvCacheManager sessionManager = new KvCacheManager(1, 4, 4, DType.F32,
                new KvBufferCacheSettings(true).withBlockSize(2), allocator, metricRegistry);
        KvBlockKey oldEpoch = tpKey(0, 2, 0, 100L, 1, 0L, 10L, 2, 1, 4, KvBlockLayout.DENSE);
        KvBlockKey newEpoch = tpKey(0, 2, 0, 101L, 1, 0L, 10L, 2, 1, 4, KvBlockLayout.DENSE);

        try (KvCacheSession session = sessionManager.openSession()) {
            writePosition(session, 0, 10.0f, 1);
            writePosition(session, 1, 20.0f, 1);
            try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
                writer.advanceLength(2);
            }
            KvBlock block = session.detachCommittedBlock(0);
            session.attachCommittedBlock(blockManager.admitAndRetain(oldEpoch, block, session.sessionId()));

            KvBlockLease oldLease = blockManager.retain(oldEpoch, session.sessionId());
            assertNotNull(oldLease);
            oldLease.close();
            assertNull(blockManager.retain(newEpoch, session.sessionId()));
        }
        blockManager.close();
    }

    @Test
    void leasedBlockSurvivesEvictionUntilSessionCloses() {
        KvBlockManager blockManager = new KvBlockManager(metricRegistry);
        KvCacheManager sessionManager = new KvCacheManager(1, 4, 4, DType.F32,
                new KvBufferCacheSettings(true).withBlockSize(2), allocator, metricRegistry);
        KvBlockKey key = localKey(0, 0L, 10L, 2, 1, 4, KvBlockLayout.DENSE);
        KvBlock block;

        try (KvCacheSession first = sessionManager.openSession()) {
            writePosition(first, 0, 10.0f, 1);
            writePosition(first, 1, 20.0f, 1);
            try (KvWriteCursor writer = first.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
                writer.advanceLength(2);
            }
            block = first.detachCommittedBlock(0);
            first.attachCommittedBlock(blockManager.admitAndRetain(key, block, first.sessionId()));
        }

        try (KvCacheSession second = sessionManager.openSession()) {
            second.attachCommittedBlock(blockManager.retain(key, second.sessionId()));
            assertTrue(blockManager.evict(key));
            assertFalse(block.isClosed());
            try (AbstractTensor keyRow = second.keyRowCopy(0, 1)) {
                assertRow(keyRow, 21.0f);
            }
        }

        assertTrue(block.isClosed());
        blockManager.close();
    }

    @Test
    void duplicateConcurrentAdmitsConvergeToOneManagedBlock() throws Exception {
        KvBlockManager blockManager = new KvBlockManager(metricRegistry);
        KvCacheManager sessionManager = new KvCacheManager(1, 4, 4, DType.F32,
                new KvBufferCacheSettings(true).withBlockSize(2), allocator, metricRegistry);
        KvBlockKey key = localKey(0, 0L, 10L, 2, 1, 4, KvBlockLayout.DENSE);
        KvBlock firstBlock = detachedSingleBlock(sessionManager, 10.0f);
        KvBlock secondBlock = detachedSingleBlock(sessionManager, 10.0f);
        CountDownLatch ready = new CountDownLatch(2);
        CountDownLatch start = new CountDownLatch(1);
        AtomicReference<KvBlockLease> firstLease = new AtomicReference<>();
        AtomicReference<KvBlockLease> secondLease = new AtomicReference<>();
        AtomicReference<Throwable> failure = new AtomicReference<>();

        Thread first = new Thread(() -> admitAfterBarrier(blockManager, key, firstBlock, "race-1", ready, start,
                firstLease, failure));
        Thread second = new Thread(() -> admitAfterBarrier(blockManager, key, secondBlock, "race-2", ready, start,
                secondLease, failure));
        first.start();
        second.start();
        ready.await();
        start.countDown();
        first.join();
        second.join();

        if (failure.get() != null) {
            throw new AssertionError(failure.get());
        }
        assertNotNull(firstLease.get());
        assertNotNull(secondLease.get());
        assertEquals(1, blockManager.residentBlockCount());
        assertEquals(2, blockManager.refCount(key));
        assertEquals(firstLease.get().blockIdentity(), secondLease.get().blockIdentity());

        firstLease.get().close();
        secondLease.get().close();
        assertEquals(0, blockManager.refCount(key));
        blockManager.close();
    }

    @Test
    void blockManagerEvictsOldestUnreferencedBlockToStayWithinByteBudget() {
        KvBlockManager blockManager = new KvBlockManager(metricRegistry, 64);
        KvCacheManager sessionManager = new KvCacheManager(1, 4, 4, DType.F32,
                new KvBufferCacheSettings(true).withBlockSize(2), allocator, metricRegistry);
        KvBlockKey firstKey = localKey(0, 0L, 10L, 2, 1, 4, KvBlockLayout.DENSE);
        KvBlockKey secondKey = localKey(0, 10L, 20L, 2, 1, 4, KvBlockLayout.DENSE);

        try (KvCacheSession first = sessionManager.openSession()) {
            KvBlock firstBlock = detachedSingleBlock(sessionManager, 10.0f);
            first.attachCommittedBlock(blockManager.admitAndRetain(firstKey, firstBlock, first.sessionId()));
        }
        assertEquals(64, blockManager.residentEncodedBytes());
        assertEquals(0, blockManager.referencedEncodedBytes());

        try (KvCacheSession second = sessionManager.openSession()) {
            KvBlock secondBlock = detachedSingleBlock(sessionManager, 30.0f);
            second.attachCommittedBlock(blockManager.admitAndRetain(secondKey, secondBlock, second.sessionId()));

            assertEquals(1, blockManager.residentBlockCount());
            assertEquals(64, blockManager.residentEncodedBytes());
            assertNull(blockManager.retain(firstKey, second.sessionId()));
            KvBlockLease secondLease = blockManager.retain(secondKey, second.sessionId());
            assertNotNull(secondLease);
            secondLease.close();
        }
        blockManager.close();
    }

    @Test
    void referencedBlocksCanTemporarilyExceedBudgetAndEvictAfterRelease() {
        KvBlockManager blockManager = new KvBlockManager(metricRegistry, 64);
        KvCacheManager sessionManager = new KvCacheManager(1, 6, 4, DType.F32,
                new KvBufferCacheSettings(true).withBlockSize(2), allocator, metricRegistry);
        KvBlockKey firstKey = localKey(0, 0L, 10L, 2, 1, 4, KvBlockLayout.DENSE);
        KvBlockKey secondKey = localKey(0, 10L, 20L, 2, 1, 4, KvBlockLayout.DENSE);
        KvBlock firstBlock = detachedSingleBlock(sessionManager, 10.0f);
        KvBlock secondBlock = detachedSingleBlock(sessionManager, 30.0f);

        try (KvCacheSession first = sessionManager.openSession();
             KvCacheSession second = sessionManager.openSession()) {
            first.attachCommittedBlock(blockManager.admitAndRetain(firstKey, firstBlock, first.sessionId()));
            second.attachCommittedBlock(blockManager.admitAndRetain(secondKey, secondBlock, second.sessionId()));

            assertEquals(2, blockManager.residentBlockCount());
            assertEquals(128, blockManager.residentEncodedBytes());
            assertEquals(128, blockManager.referencedEncodedBytes());

            first.close();

            assertTrue(firstBlock.isClosed());
            assertEquals(1, blockManager.residentBlockCount());
            assertEquals(64, blockManager.residentEncodedBytes());
            assertNull(blockManager.retain(firstKey, second.sessionId()));
            KvBlockLease secondLease = blockManager.retain(secondKey, second.sessionId());
            assertNotNull(secondLease);
            secondLease.close();
        }
        blockManager.close();
    }

    @Test
    void settingsExposeSharedPrefixBlockCacheByteBudget() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withPrefixCacheMode(KvBufferCacheSettings.PrefixCacheMode.SHARED_BLOCKS)
                .withSharedPrefixBlockCacheMaxBytes(1234L);

        assertEquals(KvBufferCacheSettings.PrefixCacheMode.SHARED_BLOCKS, settings.getPrefixCacheMode());
        assertEquals(1234L, settings.getSharedPrefixBlockCacheMaxBytes());
        assertEquals(1234L, new KvBlockManager(metricRegistry, settings).maxResidentBytes());
        assertThrows(IllegalArgumentException.class, () -> settings.setSharedPrefixBlockCacheMaxBytes(-1L));
        assertThrows(IllegalArgumentException.class, () -> settings.setPrefixCacheMode(null));
    }

    private void writePosition(KvCacheSession session, int position, float base) {
        writePosition(session, position, base, 2);
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

    private void writeFourTokenPrompt(KvCacheSession session) {
        for (int position = 0; position < 4; position++) {
            writePosition(session, position, (position + 1) * 10.0f, 1);
        }
        try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
            writer.advanceLength(4);
        }
    }

    private List<KvBlockKey> blockKeys(KvCacheSession session, long[] tokenHashes) {
        ArrayList<KvBlockKey> keys = new ArrayList<>();
        long parent = 0L;
        for (KvBlock block : session.committedBlocks()) {
            long hash = tokenHashes[block.blockIndex()];
            keys.add(localKey(block.blockIndex(), parent, hash, block.blockSize(), 1, 4, block.layout()));
            parent = hash;
        }
        return keys;
    }

    private List<Integer> transferCommittedBlocksToManager(KvCacheSession session, KvBlockManager blockManager,
            List<KvBlockKey> keys) {
        ArrayList<Integer> identities = new ArrayList<>();
        for (KvBlockKey key : keys) {
            KvBlock block = session.detachCommittedBlock(key.blockIndex());
            KvBlockLease lease = blockManager.admitAndRetain(key, block, session.sessionId());
            identities.add(lease.blockIdentity());
            session.attachCommittedBlock(lease);
        }
        return identities;
    }

    private KvBlock detachedSingleBlock(KvCacheManager manager, float base) {
        try (KvCacheSession session = manager.openSession()) {
            writePosition(session, 0, base, 1);
            writePosition(session, 1, base + 10.0f, 1);
            try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
                writer.advanceLength(2);
            }
            return session.detachCommittedBlock(0);
        }
    }

    private void admitAfterBarrier(KvBlockManager blockManager, KvBlockKey key, KvBlock block, String sessionId,
            CountDownLatch ready, CountDownLatch start, AtomicReference<KvBlockLease> lease,
            AtomicReference<Throwable> failure) {
        try {
            ready.countDown();
            start.await();
            lease.set(blockManager.admitAndRetain(key, block, sessionId));
        } catch (Throwable t) {
            failure.set(t);
        }
    }

    private KvBlockKey localKey(int blockIndex, long parentHash, long blockHash, int blockSize, int layers, int kvLength,
            KvBlockLayout layout) {
        return KvBlockKey.local("test-model", "test-runtime", blockIndex, parentHash, blockHash, blockSize,
                blockSize, layers, kvLength, DType.F32, DType.F32, layout, 0);
    }

    private KvBlockKey tpKey(int blockIndex, int tpSize, int tpRank, long epoch, int layers, long parentHash,
            long blockHash, int blockSize, int localLayers, int kvLength, KvBlockLayout layout) {
        return new KvBlockKey(1, "test-model", "none", "test-tokenizer", "test-runtime", "test-rope",
                "test-attention", blockIndex, parentHash, blockHash, blockSize,
                blockSize, localLayers, kvLength, DType.F32, DType.F32, layout, 0, tpSize, tpRank, epoch,
                "kvLength=" + kvLength + ":rank=" + tpRank + ":layers=" + layers);
    }

    private AbstractTensor row(float firstValue) {
        AbstractTensor tensor = allocator.getDirty(DType.F32, TensorShape.of(1, 4));
        for (int i = 0; i < 4; i++) {
            tensor.set(firstValue + i, 0, i);
        }
        return tensor;
    }

    private AbstractTensor wideRow(float firstValue, int width) {
        AbstractTensor tensor = allocator.getDirty(DType.F32, TensorShape.of(1, width));
        for (int i = 0; i < width; i++) {
            tensor.set(firstValue + (i % 17) * 0.125f, 0, i);
        }
        return tensor;
    }

    private AbstractTensor statisticalRow(int layer, int position, int keyOrValue, int kvLength,
            float[][][][] expected) {
        AbstractTensor tensor = allocator.getDirty(DType.F32, TensorShape.of(1, kvLength));
        for (int i = 0; i < kvLength; i++) {
            float value = (float) (Math.sin((layer + 1) * (i + 1) * 0.13)
                    + Math.cos((position + 1) * (i + 3) * 0.07)
                    + keyOrValue * 0.25
                    + layer * 0.5
                    + position * 0.125);
            tensor.set(value, 0, i);
            expected[layer][keyOrValue][position][i] = value;
        }
        return tensor;
    }

    private static double square(double value) {
        return value * value;
    }

    private static void assertRow(AbstractTensor tensor, float firstValue) {
        for (int i = 0; i < 4; i++) {
            assertEquals(firstValue + i, tensor.get(0, i), 0.0f, "col=" + i);
        }
    }
}
