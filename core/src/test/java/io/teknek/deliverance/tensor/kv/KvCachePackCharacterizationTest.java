package io.teknek.deliverance.tensor.kv;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.AbstractTensorUtils;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensor.operations.TensorOperations;
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

    @Test
    void qwenSizedI8KvSnapshotCharacterization() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();
        try {
            I8PackResult qwen06 = i8SnapshotScenario("qwen06.i8", 28, 4, 32, 1024, 2);
            I8PackResult qwen4b = i8SnapshotScenario("qwen4b.i8", 36, 4, 32, 1024, 2);

            System.out.printf("[kv-i8-characterization] qwen06=%s%n", qwen06);
            System.out.printf("[kv-i8-characterization] qwen4b=%s%n", qwen4b);
            InferenceProfiler.printSummary("kv i8 characterization", 80);

            assertTrue(qwen06.i8Bytes() < qwen06.f32Bytes() / 2, "I8 should materially reduce KV bytes");
            assertTrue(qwen4b.i8Bytes() < qwen4b.f32Bytes() / 2, "I8 should materially reduce KV bytes");
            assertTrue(qwen06.rmse() < qwen06.standardDeviation(), "Qwen 0.6B-shaped I8 RMSE should stay within one stddev");
            assertTrue(qwen4b.rmse() < qwen4b.standardDeviation(), "Qwen 4B-shaped I8 RMSE should stay within one stddev");
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    @Test
    void qwenSizedI8DecodePagedAttentionCharacterization() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();
        try {
            DecodeResult qwen06 = decodeScenario("qwen06", 28, 4, 32, 16, 16, 64, 12);
            DecodeResult qwen4b = decodeScenario("qwen4b", 36, 4, 32, 32, 8, 128, 8);

            System.out.printf("[kv-i8-decode-characterization] qwen06=%s%n", qwen06);
            System.out.printf("[kv-i8-decode-characterization] qwen4b=%s%n", qwen4b);
            InferenceProfiler.printSummary("kv i8 decode characterization", 80);

            assertTrue(qwen06.bf16Bytes() < qwen06.f32Bytes(), "BF16 should reduce KV bytes");
            assertTrue(qwen4b.bf16Bytes() < qwen4b.f32Bytes(), "BF16 should reduce KV bytes");
            assertTrue(qwen06.i8Bytes() < qwen06.bf16Bytes(), "I8 should reduce KV bytes more than BF16");
            assertTrue(qwen4b.i8Bytes() < qwen4b.bf16Bytes(), "I8 should reduce KV bytes more than BF16");
            assertTrue(qwen06.i8Rmse() < qwen06.outputStddev(), "Qwen 0.6B-shaped I8 decode RMSE should stay within output stddev");
            assertTrue(qwen4b.i8Rmse() < qwen4b.outputStddev(), "Qwen 4B-shaped I8 decode RMSE should stay within output stddev");
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    @Test
    void qwenSizedKvWriteDtypeCharacterization() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();
        try (io.teknek.deliverance.math.WrappedForkJoinPool pool = new io.teknek.deliverance.math.WrappedForkJoinPool(
                new java.util.concurrent.ForkJoinPool(4))) {
            TensorOperations provider = new NativeSimdTensorOperations(new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                    allocator, pool));
            WriteResult f32 = writeDTypeScenario("qwen06.f32.constructor", DType.F32, null, 28, 4, 32, 1024, 2);
            WriteResult bf16 = writeDTypeScenario("qwen06.bf16.constructor", DType.BF16, null, 28, 4, 32, 1024, 2);
            WriteResult i8 = writeDTypeScenario("qwen06.i8.constructor", DType.I8, null, 28, 4, 32, 1024, 2);
            WriteResult f32Provider = writeDTypeScenario("qwen06.f32.provider", DType.F32, provider, 28, 4, 32, 1024, 2);
            WriteResult bf16Provider = writeDTypeScenario("qwen06.bf16.provider", DType.BF16, provider, 28, 4, 32, 1024, 2);
            WriteResult i8Provider = writeDTypeScenario("qwen06.i8.provider", DType.I8, provider, 28, 4, 32, 1024, 2);

            System.out.printf("[kv-write-characterization] f32_constructor=%s%n", f32);
            System.out.printf("[kv-write-characterization] bf16_constructor=%s%n", bf16);
            System.out.printf("[kv-write-characterization] i8_constructor=%s%n", i8);
            System.out.printf("[kv-write-characterization] f32_provider=%s%n", f32Provider);
            System.out.printf("[kv-write-characterization] bf16_provider=%s%n", bf16Provider);
            System.out.printf("[kv-write-characterization] i8_provider=%s%n", i8Provider);
            InferenceProfiler.printSummary("kv write dtype characterization", 80);

            assertEquals(f32.writes(), bf16.writes());
            assertEquals(f32.writes(), i8.writes());
            assertEquals(f32.writes(), f32Provider.writes());
            assertEquals(f32.writes(), bf16Provider.writes());
            assertEquals(f32.writes(), i8Provider.writes());
            assertTrue(bf16.bytes() < f32.bytes(), "BF16 KV should use fewer dense bytes than F32");
            assertTrue(i8.bytes() < bf16.bytes(), "I8 KV should use fewer dense bytes than BF16");
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

    private I8PackResult i8SnapshotScenario(String name, int layers, int visibleBlocks, int blockSize, int kvLength,
            int repetitions) {
        int visibleRows = visibleBlocks * blockSize;
        long quantizeNanos = 0;
        long packNanos = 0;
        long squaredErrorCount = 0;
        double squaredError = 0.0;
        double sum = 0.0;
        double sumSquares = 0.0;
        AbstractTensor[] keyLayers = new AbstractTensor[layers];
        AbstractTensor[] valueLayers = new AbstractTensor[layers];
        try {
            for (int layer = 0; layer < layers; layer++) {
                try (AbstractTensor denseKeys = allocator.getDirty(DType.F32, TensorShape.of(visibleRows, kvLength));
                     AbstractTensor denseValues = allocator.getDirty(DType.F32, TensorShape.of(visibleRows, kvLength))) {
                    for (int position = 0; position < visibleRows; position++) {
                        for (int col = 0; col < kvLength; col++) {
                            float key = value(layer, position, 0, col);
                            float val = value(layer, position, 1, col);
                            denseKeys.set(key, position, col);
                            denseValues.set(val, position, col);
                            sum += key + val;
                            sumSquares += key * key + val * val;
                            squaredErrorCount += 2;
                        }
                    }
                    long start = System.nanoTime();
                    keyLayers[layer] = AbstractTensorUtils.quantize(denseKeys, DType.I8, true);
                    valueLayers[layer] = AbstractTensorUtils.quantize(denseValues, DType.I8, true);
                    quantizeNanos += System.nanoTime() - start;
                }
            }

            try (AbstractTensor packedKeys = allocator.getDirty(DType.F32, TensorShape.of(visibleRows, kvLength));
                 AbstractTensor packedValues = allocator.getDirty(DType.F32, TensorShape.of(visibleRows, kvLength))) {
                for (int repetition = 0; repetition < repetitions; repetition++) {
                    for (int layer = 0; layer < layers; layer++) {
                        long start = System.nanoTime();
                        packI8Layer(keyLayers[layer], valueLayers[layer], packedKeys, packedValues, visibleRows, kvLength);
                        packNanos += System.nanoTime() - start;
                    }
                }
            }

            for (int layer = 0; layer < layers; layer++) {
                for (int position = 0; position < visibleRows; position++) {
                    for (int col = 0; col < kvLength; col++) {
                        double expectedKey = value(layer, position, 0, col);
                        double expectedValue = value(layer, position, 1, col);
                        double actualKey = keyLayers[layer].get(position, col);
                        double actualValue = valueLayers[layer].get(position, col);
                        squaredError += square(expectedKey - actualKey) + square(expectedValue - actualValue);
                    }
                }
            }

            InferenceProfiler.timer(metricRegistry, "kvpack.characterization." + name + ".quantize")
                    .update(quantizeNanos, TimeUnit.NANOSECONDS);
            InferenceProfiler.timer(metricRegistry, "kvpack.characterization." + name + ".pack")
                    .update(packNanos, TimeUnit.NANOSECONDS);
            long f32Bytes = (long) layers * visibleRows * 2 * kvLength * DType.F32.size();
            long i8Bytes = (long) layers * visibleRows * 2 * kvLength * DType.I8.size()
                    + (long) layers * visibleRows * 2 * (kvLength / 32) * DType.F32.size();
            double mean = sum / squaredErrorCount;
            double stddev = Math.sqrt((sumSquares / squaredErrorCount) - (mean * mean));
            double rmse = Math.sqrt(squaredError / squaredErrorCount);
            return new I8PackResult(layers, visibleRows, kvLength, repetitions,
                    TimeUnit.NANOSECONDS.toMicros(quantizeNanos), TimeUnit.NANOSECONDS.toMicros(packNanos),
                    f32Bytes, i8Bytes, rmse, stddev);
        } finally {
            for (AbstractTensor tensor : keyLayers) {
                if (tensor != null) {
                    tensor.close();
                }
            }
            for (AbstractTensor tensor : valueLayers) {
                if (tensor != null) {
                    tensor.close();
                }
            }
        }
    }

    private DecodeResult decodeScenario(String name, int layers, int visibleBlocks, int blockSize, int numberOfHeads,
            int numberOfKeyValueHeads, int headSize, int repetitions) {
        int visibleRows = visibleBlocks * blockSize;
        int kvLength = numberOfKeyValueHeads * headSize;
        int attentionLength = numberOfHeads * headSize;
        TensorOperations ops = new NativeSimdTensorOperations(new PanamaTensorOperations(MachineSpec.VECTOR_TYPE,
                allocator, new io.teknek.deliverance.math.WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(4))));
        try (AbstractTensor query = allocator.getDirty(DType.F32, TensorShape.of(1, attentionLength));
             AbstractTensor f32Out = allocator.getDirty(DType.F32, TensorShape.of(1, attentionLength));
             AbstractTensor bf16Out = allocator.getDirty(DType.F32, TensorShape.of(1, attentionLength));
             AbstractTensor i8Out = allocator.getDirty(DType.F32, TensorShape.of(1, attentionLength))) {
            fillQuery(query);
            AbstractTensor[] f32Keys = new AbstractTensor[layers];
            AbstractTensor[] f32Values = new AbstractTensor[layers];
            AbstractTensor[] bf16Keys = new AbstractTensor[layers];
            AbstractTensor[] bf16Values = new AbstractTensor[layers];
            AbstractTensor[] i8Keys = new AbstractTensor[layers];
            AbstractTensor[] i8Values = new AbstractTensor[layers];
            try {
                for (int layer = 0; layer < layers; layer++) {
                    f32Keys[layer] = allocator.getDirty(DType.F32, TensorShape.of(visibleRows, kvLength));
                    f32Values[layer] = allocator.getDirty(DType.F32, TensorShape.of(visibleRows, kvLength));
                    fillKvPage(f32Keys[layer], layer, 0);
                    fillKvPage(f32Values[layer], layer, 1);
                    bf16Keys[layer] = AbstractTensorUtils.quantize(f32Keys[layer], DType.BF16, true);
                    bf16Values[layer] = AbstractTensorUtils.quantize(f32Values[layer], DType.BF16, true);
                    i8Keys[layer] = AbstractTensorUtils.quantize(f32Keys[layer], DType.I8, true);
                    i8Values[layer] = AbstractTensorUtils.quantize(f32Values[layer], DType.I8, true);
                }

                long f32Nanos = 0;
                long bf16Nanos = 0;
                long i8Nanos = 0;
                for (int repetition = 0; repetition < repetitions; repetition++) {
                    for (int layer = 0; layer < layers; layer++) {
                        f32Out.clear();
                        long f32Start = System.nanoTime();
                        ops.decodePagedAttention(f32Out, query, new AbstractTensor[]{f32Keys[layer]},
                                new AbstractTensor[]{f32Values[layer]}, visibleRows, numberOfHeads, numberOfKeyValueHeads,
                                headSize, 1.0f / (float) Math.sqrt(headSize), null);
                        f32Nanos += System.nanoTime() - f32Start;

                        bf16Out.clear();
                        long bf16Start = System.nanoTime();
                        ops.decodePagedAttention(bf16Out, query, new AbstractTensor[]{bf16Keys[layer]},
                                new AbstractTensor[]{bf16Values[layer]}, visibleRows, numberOfHeads, numberOfKeyValueHeads,
                                headSize, 1.0f / (float) Math.sqrt(headSize), null);
                        bf16Nanos += System.nanoTime() - bf16Start;

                        i8Out.clear();
                        long i8Start = System.nanoTime();
                        ops.decodePagedAttention(i8Out, query, new AbstractTensor[]{i8Keys[layer]},
                                new AbstractTensor[]{i8Values[layer]}, visibleRows, numberOfHeads, numberOfKeyValueHeads,
                                headSize, 1.0f / (float) Math.sqrt(headSize), null);
                        i8Nanos += System.nanoTime() - i8Start;
                    }
                }

                double squaredError = 0.0;
                double bf16SquaredError = 0.0;
                double sum = 0.0;
                double sumSquares = 0.0;
                int count = 0;
                for (int layer = 0; layer < layers; layer++) {
                    f32Out.clear();
                    bf16Out.clear();
                    i8Out.clear();
                    ops.decodePagedAttention(f32Out, query, new AbstractTensor[]{f32Keys[layer]},
                            new AbstractTensor[]{f32Values[layer]}, visibleRows, numberOfHeads, numberOfKeyValueHeads,
                            headSize, 1.0f / (float) Math.sqrt(headSize), null);
                    ops.decodePagedAttention(bf16Out, query, new AbstractTensor[]{bf16Keys[layer]},
                            new AbstractTensor[]{bf16Values[layer]}, visibleRows, numberOfHeads, numberOfKeyValueHeads,
                            headSize, 1.0f / (float) Math.sqrt(headSize), null);
                    ops.decodePagedAttention(i8Out, query, new AbstractTensor[]{i8Keys[layer]},
                            new AbstractTensor[]{i8Values[layer]}, visibleRows, numberOfHeads, numberOfKeyValueHeads,
                            headSize, 1.0f / (float) Math.sqrt(headSize), null);
                    for (int col = 0; col < attentionLength; col++) {
                        double expected = f32Out.get(0, col);
                        double bf16Actual = bf16Out.get(0, col);
                        double actual = i8Out.get(0, col);
                        bf16SquaredError += square(expected - bf16Actual);
                        squaredError += square(expected - actual);
                        sum += expected;
                        sumSquares += expected * expected;
                        count++;
                    }
                }
                InferenceProfiler.timer(metricRegistry, "kvdecode.characterization." + name + ".f32")
                        .update(f32Nanos, TimeUnit.NANOSECONDS);
                InferenceProfiler.timer(metricRegistry, "kvdecode.characterization." + name + ".bf16")
                        .update(bf16Nanos, TimeUnit.NANOSECONDS);
                InferenceProfiler.timer(metricRegistry, "kvdecode.characterization." + name + ".i8")
                        .update(i8Nanos, TimeUnit.NANOSECONDS);
                long f32Bytes = denseKvBytes(layers, visibleRows, kvLength, DType.F32);
                long bf16Bytes = denseKvBytes(layers, visibleRows, kvLength, DType.BF16);
                long i8Bytes = denseKvBytes(layers, visibleRows, kvLength, DType.I8);
                double mean = sum / count;
                double stddev = Math.sqrt((sumSquares / count) - (mean * mean));
                return new DecodeResult(layers, visibleRows, attentionLength, kvLength, repetitions,
                        TimeUnit.NANOSECONDS.toMicros(f32Nanos), TimeUnit.NANOSECONDS.toMicros(bf16Nanos),
                        TimeUnit.NANOSECONDS.toMicros(i8Nanos), f32Bytes, bf16Bytes, i8Bytes,
                        Math.sqrt(bf16SquaredError / count), Math.sqrt(squaredError / count), stddev);
            } finally {
                closeAll(f32Keys);
                closeAll(f32Values);
                closeAll(bf16Keys);
                closeAll(bf16Values);
                closeAll(i8Keys);
                closeAll(i8Values);
            }
        }
    }

    private WriteResult writeDTypeScenario(String name, DType dtype, TensorOperations provider, int layers,
            int visibleBlocks, int blockSize, int kvLength, int repetitions) {
        int visibleRows = visibleBlocks * blockSize;
        long elapsedNanos = 0;
        int writes = 0;
        for (int repetition = 0; repetition < repetitions; repetition++) {
            KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                    .withBlockSize(blockSize)
                    .withKvKeyDType(dtype)
                    .withKvValueDType(dtype);
            KvCacheManager manager = provider == null
                    ? new KvCacheManager(layers, visibleRows, kvLength, DType.F32, settings, allocator,
                            metricRegistry)
                    : new KvCacheManager(layers, visibleRows, kvLength, DType.F32, settings, allocator,
                            metricRegistry, false, provider);
            try (KvCacheSession session = manager.openSession();
                 KvWriteCursor writer = session.writer(CacheExecutionMode.PREFILL_UPDATE_CACHE)) {
                for (int position = 0; position < visibleRows; position++) {
                    for (int layer = 0; layer < layers; layer++) {
                        try (AbstractTensor key = row(layer, position, 0, kvLength);
                             AbstractTensor value = row(layer, position, 1, kvLength)) {
                            long start = System.nanoTime();
                            writer.write(layer, position, key, value);
                            elapsedNanos += System.nanoTime() - start;
                            writes++;
                        }
                    }
                    writer.advanceLength(position + 1);
                }
            }
        }
        InferenceProfiler.timer(metricRegistry, "kvwrite.characterization." + name)
                .update(elapsedNanos, TimeUnit.NANOSECONDS);
        long bytes = denseKvBytes(layers, visibleRows, kvLength, dtype);
        return new WriteResult(dtype, layers, visibleRows, kvLength, repetitions, writes,
                TimeUnit.NANOSECONDS.toMicros(elapsedNanos), bytes);
    }

    private void fillQuery(AbstractTensor query) {
        for (int col = 0; col < query.shape().last(); col++) {
            query.set((float) (Math.sin((col + 1) * 0.017) + Math.cos((col + 3) * 0.011)), 0, col);
        }
    }

    private void fillKvPage(AbstractTensor page, int layer, int keyOrValue) {
        for (int row = 0; row < page.shape().first(); row++) {
            for (int col = 0; col < page.shape().last(); col++) {
                page.set(value(layer, row, keyOrValue, col), row, col);
            }
        }
    }

    private void closeAll(AbstractTensor[] tensors) {
        for (AbstractTensor tensor : tensors) {
            if (tensor != null) {
                tensor.close();
            }
        }
    }

    private void packI8Layer(AbstractTensor keys, AbstractTensor values, AbstractTensor packedKeys,
            AbstractTensor packedValues, int visibleRows, int kvLength) {
        for (int position = 0; position < visibleRows; position++) {
            try (AbstractTensor keyRow = keys.slice(position);
                 AbstractTensor valueRow = values.slice(position)) {
                for (int col = 0; col < kvLength; col++) {
                    packedKeys.set(keyRow.get(0, col), position, col);
                    packedValues.set(valueRow.get(0, col), position, col);
                }
            }
        }
    }

    private static double square(double value) {
        return value * value;
    }

    private static long denseKvBytes(int layers, int visibleRows, int kvLength, DType dtype) {
        long dataBytes = (long) layers * visibleRows * 2 * kvLength * dtype.size();
        if (dtype == DType.I8) {
            return dataBytes + (long) layers * visibleRows * 2 * (kvLength / 32) * DType.F32.size();
        }
        return dataBytes;
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

    private record I8PackResult(int layers, int visibleRows, int kvLength, int repetitions, long quantizeMicros,
            long packMicros, long f32Bytes, long i8Bytes, double rmse, double standardDeviation) {
    }

    private record WriteResult(DType dtype, int layers, int visibleRows, int kvLength, int repetitions, int writes,
            long elapsedMicros, long bytes) {
    }

    private record DecodeResult(int layers, int visibleRows, int attentionLength, int kvLength, int repetitions,
            long f32Micros, long bf16Micros, long i8Micros, long f32Bytes, long bf16Bytes, long i8Bytes,
            double bf16Rmse, double i8Rmse, double outputStddev) {
    }
}
