package io.teknek.deliverance.tensor.kv;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Characterizes KVCache2 visible-row packing on model-scale synthetic KV rows without loading model weights. */
class KvCachePackCharacterizationTest {
    private static final int NEMOTRON_LAYERS = 26;
    private static final int NEMOTRON_BLOCK_SIZE = 32;
    private static final int NEMOTRON_KV_LENGTH = 1024;

    private final MetricRegistry metricRegistry = new MetricRegistry();
    private final TensorAllocator allocator = new ArrayQueueTensorAllocator(metricRegistry);

    @Test
    void tinyDenseAndTurboQuantPackingHaveExpectedShapesAndCompression() {
        PackResult dense = packScenario("tiny.dense", KvBufferCacheSettings.KvBlockStoragePolicy.DENSE,
                2, 2, 4, 64, 1);
        PackResult turbo = packScenario("tiny.turboquant", KvBufferCacheSettings.KvBlockStoragePolicy.MSE_TURBOQUANT,
                2, 2, 4, 64, 1);

        assertEquals(8, dense.visibleRows());
        assertEquals(8, turbo.visibleRows());
        assertEquals(0, dense.encodedBytes());
        assertTrue(turbo.encodedBytes() > 0);
        assertTrue(turbo.encodedBytes() < turbo.denseBytesEquivalent() / 2,
                "expected compressed KV block storage to be materially smaller");
    }

    @Test
    void nemotronSizedDenseAndTurboQuantPackCharacterization() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();
        try {
            PackResult dense = packScenario("nemotron.dense", KvBufferCacheSettings.KvBlockStoragePolicy.DENSE,
                    NEMOTRON_LAYERS, 4, NEMOTRON_BLOCK_SIZE, NEMOTRON_KV_LENGTH, 3);
            PackResult turbo = packScenario("nemotron.turboquant", KvBufferCacheSettings.KvBlockStoragePolicy.MSE_TURBOQUANT,
                    NEMOTRON_LAYERS, 4, NEMOTRON_BLOCK_SIZE, NEMOTRON_KV_LENGTH, 3);

            System.out.printf("[kv-pack-characterization] dense=%s%n", dense);
            System.out.printf("[kv-pack-characterization] turbo=%s%n", turbo);
            System.out.printf("[kv-pack-characterization] turbo_encoded_ratio=%.4f%n",
                    turbo.encodedBytes() / (double) turbo.denseBytesEquivalent());
            InferenceProfiler.printSummary("kv pack characterization", 60);

            assertEquals(dense.visibleRows(), turbo.visibleRows());
            assertTrue(turbo.encodedBytes() < turbo.denseBytesEquivalent() / 2,
                    "expected compressed KV storage to be materially smaller");
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    @Disabled("Manual larger sweep for KV pack optimization work; enable when comparing implementations.")
    @Test
    void nemotronSizedVisibleLengthSweep() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();
        try {
            int[] blocks = {1, 2, 4, 8};
            for (int visibleBlocks : blocks) {
                PackResult dense = packScenario("sweep.dense.blocks." + visibleBlocks,
                        KvBufferCacheSettings.KvBlockStoragePolicy.DENSE, NEMOTRON_LAYERS, visibleBlocks,
                        NEMOTRON_BLOCK_SIZE, NEMOTRON_KV_LENGTH, 2);
                PackResult turbo = packScenario("sweep.turbo.blocks." + visibleBlocks,
                        KvBufferCacheSettings.KvBlockStoragePolicy.MSE_TURBOQUANT, NEMOTRON_LAYERS, visibleBlocks,
                        NEMOTRON_BLOCK_SIZE, NEMOTRON_KV_LENGTH, 2);
                System.out.printf("[kv-pack-characterization] visible_blocks=%d dense=%s turbo=%s%n",
                        visibleBlocks, dense, turbo);
            }
            InferenceProfiler.printSummary("kv pack characterization sweep", 80);
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    private PackResult packScenario(String name, KvBufferCacheSettings.KvBlockStoragePolicy policy, int layers,
            int visibleBlocks, int blockSize, int kvLength, int repetitions) {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withBlockSize(blockSize)
                .withKvBlockStoragePolicy(policy)
                .withKvTurboQuantBits(4);
        KvCacheManager manager = new KvCacheManager(layers, visibleBlocks * blockSize, kvLength, DType.F32,
                settings, allocator, metricRegistry);
        long beforeDenseBytes = metricRegistry.counter("kvcache.v2.turboquant.dense.bytes.equivalent").getCount();
        long beforeEncodedBytes = metricRegistry.counter("kvcache.v2.turboquant.encoded.bytes").getCount();

        try (KvCacheSession session = manager.openSession()) {
            fillCommittedBlocks(session, layers, visibleBlocks, blockSize, kvLength);
            int visibleRows = visibleBlocks * blockSize;
            long elapsedNanos = 0;
            for (int repetition = 0; repetition < repetitions; repetition++) {
                for (int layer = 0; layer < layers; layer++) {
                    try (KvReadView readView = session.readView(layer, visibleRows, AttentionPattern.CAUSAL);
                         AbstractTensor packedKeys = allocator.getDirty(DType.F32, TensorShape.of(visibleRows, kvLength));
                         AbstractTensor packedValues = allocator.getDirty(DType.F32, TensorShape.of(visibleRows, kvLength))) {
                        long start = System.nanoTime();
                        packVisibleRows(readView, packedKeys, packedValues, visibleRows, kvLength);
                        elapsedNanos += System.nanoTime() - start;
                        assertEquals(visibleRows, packedKeys.shape().first());
                        assertEquals(kvLength, packedValues.shape().last());
                    }
                }
            }
            InferenceProfiler.timer(metricRegistry, "kvpack.characterization." + name)
                    .update(elapsedNanos, TimeUnit.NANOSECONDS);
            return new PackResult(policy, layers, visibleRows, kvLength, repetitions,
                    TimeUnit.NANOSECONDS.toMicros(elapsedNanos),
                    metricRegistry.counter("kvcache.v2.turboquant.dense.bytes.equivalent").getCount() - beforeDenseBytes,
                    metricRegistry.counter("kvcache.v2.turboquant.encoded.bytes").getCount() - beforeEncodedBytes);
        }
    }

    private void fillCommittedBlocks(KvCacheSession session, int layers, int visibleBlocks, int blockSize, int kvLength) {
        try (KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
            for (int position = 0; position < visibleBlocks * blockSize; position++) {
                for (int layer = 0; layer < layers; layer++) {
                    try (AbstractTensor key = row(layer, position, 0, kvLength);
                         AbstractTensor value = row(layer, position, 1, kvLength)) {
                        writer.write(layer, position, key, value);
                    }
                }
                writer.advanceLength(position + 1);
            }
        }
    }

    private void packVisibleRows(KvReadView readView, AbstractTensor packedKeys, AbstractTensor packedValues,
            int visibleRows, int kvLength) {
        readView.copyKeyRows(0, visibleRows, packedKeys, 0);
        readView.copyValueRows(0, visibleRows, packedValues, 0);
    }

    private AbstractTensor row(int layer, int position, int keyOrValue, int kvLength) {
        AbstractTensor tensor = allocator.getDirty(DType.F32, TensorShape.of(1, kvLength));
        for (int i = 0; i < kvLength; i++) {
            tensor.set(value(layer, position, keyOrValue, i), 0, i);
        }
        return tensor;
    }

    private static float value(int layer, int position, int keyOrValue, int index) {
        return (float) (Math.sin((layer + 1) * (index + 1) * 0.013)
                + Math.cos((position + 1) * (index + 3) * 0.007)
                + keyOrValue * 0.25
                + (layer % 7) * 0.03125
                + (position % 11) * 0.015625);
    }

    private record PackResult(KvBufferCacheSettings.KvBlockStoragePolicy policy, int layers, int visibleRows,
            int kvLength, int repetitions, long elapsedMicros, long denseBytesEquivalent, long encodedBytes) {
    }
}
