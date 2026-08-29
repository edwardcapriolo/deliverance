package io.teknek.deliverance.tensor.kv;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.ReadOnlyTensor;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.TrackedReadOnlyTensor;
import org.junit.jupiter.api.Test;

import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

    private AbstractTensor row(float firstValue) {
        AbstractTensor tensor = allocator.getDirty(DType.F32, TensorShape.of(1, 4));
        for (int i = 0; i < 4; i++) {
            tensor.set(firstValue + i, 0, i);
        }
        return tensor;
    }

    private static void assertRow(AbstractTensor tensor, float firstValue) {
        for (int i = 0; i < 4; i++) {
            assertEquals(firstValue + i, tensor.get(0, i), 0.0f, "col=" + i);
        }
    }
}
