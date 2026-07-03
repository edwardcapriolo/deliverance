package io.teknek.deliverance.tensor;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class KvBufferCachePrefixTest {

    private AbstractModel mockModel() {
        Config config = new Config(128, 64, 128, 4,
                4, 2, 1e-5f, 1000, 0,
                List.of(1), ActivationFunction.Type.GELU_PYTORCH_TANH, null, null);
        return mockModel(config);
    }

    private AbstractModel mockModel(Config config) {
        AbstractModel model = mock(AbstractModel.class);
        when(model.getConfig()).thenReturn(config);
        when(model.getLocalKvLength()).thenReturn(config.kvLength);
        when(model.getWorkingDType()).thenReturn(DType.F32);
        when(model.getTensorAllocator()).thenReturn(new ArrayQueueTensorAllocator(new MetricRegistry()));
        when(model.getMetricRegistry()).thenReturn(new MetricRegistry());
        return model;
    }

    @Test
    public void computePageSizeUsesContextRowsTargetForAttentionLocality() {
        Config qwen3FourBShape = new Config(40960, 2560, 9728, 32,
                8, 36, 1e-6f, 151936, 151643,
                List.of(151645), ActivationFunction.Type.SILU, null, null,
                128, null, null);
        KvBufferCache cache = new KvBufferCache(mockModel(qwen3FourBShape), new KvBufferCacheSettings(true));
        KvBufferCache.KvBuffer buffer = cache.new KvBuffer("qwen3-4b-shape", 1 << 20);

        KvBufferCache.KvPageContext page = buffer.computePageSize(1 << 20);

        assertEquals(4, page.layersPerPage());
        assertEquals(32, page.contextLengthPerPage());
    }

    @Test
    public void testExactPrefixHit() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withMaxEntries(512)
                .withBlockSize(4)
                .withPrefixCheckpointPolicy(KvBufferCacheSettings.PrefixCheckpointPolicy.FIXED_BLOCKS);
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);
        int[] tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        KvBufferCache.KvBuffer buf = cache.getEphemeralKvBuffer();
        cache.storePrefix(tokens, buf, Optional.empty());

        {
            int[] tokens2 = {1, 2, 3};
            KvBufferCache.KvBuffer buf2 = cache.getEphemeralKvBuffer();
            cache.storePrefix(tokens2, buf2, Optional.empty());
        }

        KvBufferCache.PrefixEntry e = cache.lookupPrefix(tokens, Optional.empty());
        assertEquals(8, e.length());
    }

    @Test
    public void testDisabled() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true).withMaxEntries(0);
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);
        int[] tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        KvBufferCache.KvBuffer buf = cache.getEphemeralKvBuffer();
        cache.storePrefix(tokens, buf, Optional.empty());
        assertNull(cache.lookupPrefix(tokens, Optional.empty()));
    }

    @Test
    public void testSaltyDisabled() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withMaxEntries(10)
                .withBlockSize(4)
                .withPrefixCheckpointPolicy(KvBufferCacheSettings.PrefixCheckpointPolicy.FIXED_BLOCKS);
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);
        int[] tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        KvBufferCache.KvBuffer buf = cache.getEphemeralKvBuffer();
        cache.storePrefix(tokens, buf, Optional.of("stillsalty"));

        //second
        KvBufferCache.KvBuffer buf2 = cache.getEphemeralKvBuffer();
        cache.storePrefix(tokens, buf2, Optional.of("yolo"));

        assertNotSame(cache.lookupPrefix(tokens, Optional.of("yolo")),
                cache.lookupPrefix(tokens, Optional.of("stillsalty")));
    }

    @Test
    public void keyRowsRoundTripThroughPagesInOrder() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true).withBlockSize(1);
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);
        KvBufferCache.KvBuffer buffer = cache.new KvBuffer("test", 1024);

        for (int pos = 0; pos < 5; pos++) {
            try (AbstractTensor row = buffer.getKeyTensorForPosition(0, pos)) {
                row.set(pos * 10 + 1, 0, 0);
                row.set(pos * 10 + 2, 0, 1);
                row.set(pos * 10 + 3, 0, 2);
                row.set(pos * 10 + 4, 0, 3);
            }
        }


        try (AbstractTensor packed = new FloatBufferTensor(5, 4)) {
            AbstractTensor[] pages = buffer.getKeyTensorsUptoPosition(0, 4);
            try {
                assertEquals(5, fillVisibleRows(packed, pages, 4, 0, 4));
                assertEquals(1.0f, packed.get(0, 0));
                assertEquals(2.0f, packed.get(0, 1));
                assertEquals(3.0f, packed.get(0, 2));
                assertEquals(4.0f, packed.get(0, 3));

                assertEquals(11.0f, packed.get(1, 0));
                assertEquals(12.0f, packed.get(1, 1));
                assertEquals(13.0f, packed.get(1, 2));
                assertEquals(14.0f, packed.get(1, 3));

                assertEquals(21.0f, packed.get(2, 0));
                assertEquals(22.0f, packed.get(2, 1));
                assertEquals(23.0f, packed.get(2, 2));
                assertEquals(24.0f, packed.get(2, 3));

                assertEquals(31.0f, packed.get(3, 0));
                assertEquals(32.0f, packed.get(3, 1));
                assertEquals(33.0f, packed.get(3, 2));
                assertEquals(34.0f, packed.get(3, 3));

                assertEquals(41.0f, packed.get(4, 0));
                assertEquals(42.0f, packed.get(4, 1));
                assertEquals(43.0f, packed.get(4, 2));
                assertEquals(44.0f, packed.get(4, 3));
            } finally {
                for (AbstractTensor page : pages) {
                    page.close();
                }
                buffer.close();
            }
        }
    }

    @Test
    public void storeLookupAndCopyPrefixPreservesKeysAndValues() {
        int expectedPrefixLength = 8;
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withMaxEntries(512)
                .withBlockSize(4)
                .withPrefixCheckpointPolicy(KvBufferCacheSettings.PrefixCheckpointPolicy.FIXED_BLOCKS);
        AbstractModel model = mockModel();
        KvBufferCache cache = new KvBufferCache(model, settings);
        int[] tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};

        try (KvBufferCache.KvBuffer source = cache.getEphemeralKvBuffer();
             KvBufferCache.KvBuffer copied = cache.getEphemeralKvBuffer()) {
            fillKv(source, model.getConfig(), expectedPrefixLength);

            cache.storePrefix(tokens, source, Optional.empty());
            KvBufferCache.PrefixEntry hit = cache.lookupPrefix(tokens, Optional.empty());

            assertNotNull(hit);
            assertEquals(expectedPrefixLength, hit.length());

            cache.copyPrefix(hit.buffer(), copied, hit.length());
            assertKvPrefixEquals(source, copied, model.getConfig(), hit.length());
        }
    }

    @Test
    public void lz4PrefixStoreLookupAndCopyPreservesKeysAndValues() {
        int expectedPrefixLength = 8;
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withMaxEntries(512)
                .withBlockSize(4)
                .withPrefixCheckpointPolicy(KvBufferCacheSettings.PrefixCheckpointPolicy.FIXED_BLOCKS)
                .withPrefixCompression(KvBufferCacheSettings.PrefixCompression.LZ4);
        AbstractModel model = mockModel();
        KvBufferCache cache = new KvBufferCache(model, settings);
        int[] tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};

        try (KvBufferCache.KvBuffer source = cache.getEphemeralKvBuffer();
             KvBufferCache.KvBuffer copied = cache.getEphemeralKvBuffer()) {
            fillKv(source, model.getConfig(), expectedPrefixLength);

            cache.storePrefix(tokens, source, Optional.empty());
            KvBufferCache.PrefixEntry hit = cache.lookupPrefix(tokens, Optional.empty());

            assertNotNull(hit);
            assertTrue(hit.temporary(), "LZ4 lookup hydrates a temporary KV buffer");
            assertEquals(expectedPrefixLength, hit.length());

            try {
                cache.copyPrefix(hit.buffer(), copied, hit.length());
            } finally {
                hit.closeIfTemporary();
            }
            assertKvPrefixEquals(source, copied, model.getConfig(), hit.length());
        }
    }

    @Test
    public void lz4PrefixCompressionShrinksZeroKvSnapshots() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withMaxEntries(512)
                .withBlockSize(4)
                .withPrefixCheckpointPolicy(KvBufferCacheSettings.PrefixCheckpointPolicy.FIXED_BLOCKS)
                .withPrefixCompression(KvBufferCacheSettings.PrefixCompression.LZ4);
        AbstractModel model = mockModel();
        KvBufferCache cache = new KvBufferCache(model, settings);
        int[] tokens = {1, 2, 3, 4};

        try (KvBufferCache.KvBuffer source = cache.getEphemeralKvBuffer()) {
            cache.storePrefix(tokens, source, Optional.empty());

            long uncompressedBytes = model.getMetricRegistry()
                    .counter("kvbuffercache.prefix.lz4.uncompressed.bytes").getCount();
            long compressedBytes = model.getMetricRegistry()
                    .counter("kvbuffercache.prefix.lz4.compressed.bytes").getCount();

            assertTrue(uncompressedBytes > 0);
            assertTrue(compressedBytes < uncompressedBytes / 10,
                    "expected zeros to compress well: compressed=" + compressedBytes
                            + " uncompressed=" + uncompressedBytes);
        }
    }

    @Test
    public void mseTurboQuantPrefixStoreLookupAndCopyApproximatelyPreservesKeysAndValues() {
        int expectedPrefixLength = 8;
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withMaxEntries(512)
                .withBlockSize(4)
                .withPrefixCheckpointPolicy(KvBufferCacheSettings.PrefixCheckpointPolicy.FIXED_BLOCKS)
                .withPrefixCompression(KvBufferCacheSettings.PrefixCompression.MSE_TURBOQUANT)
                .withPrefixTurboQuantBits(4);
        AbstractModel model = mockModel();
        KvBufferCache cache = new KvBufferCache(model, settings);
        int[] tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};

        try (KvBufferCache.KvBuffer source = cache.getEphemeralKvBuffer();
             KvBufferCache.KvBuffer copied = cache.getEphemeralKvBuffer()) {
            fillKv(source, model.getConfig(), expectedPrefixLength);

            cache.storePrefix(tokens, source, Optional.empty());
            KvBufferCache.PrefixEntry hit = cache.lookupPrefix(tokens, Optional.empty());

            assertNotNull(hit);
            assertTrue(hit.temporary(), "MSE TurboQuant lookup hydrates a temporary KV buffer");
            assertEquals(expectedPrefixLength, hit.length());
            try {
                cache.copyPrefix(hit.buffer(), copied, hit.length());
            } finally {
                hit.closeIfTemporary();
            }

            assertKvPrefixRmseBelow(source, copied, model.getConfig(), hit.length(), 275.0f);
            long rawBytes = model.getMetricRegistry()
                    .counter("kvbuffercache.prefix.turboquant.raw.bytes").getCount();
            long encodedBytes = model.getMetricRegistry()
                    .counter("kvbuffercache.prefix.turboquant.encoded.bytes").getCount();
            assertTrue(rawBytes > 0);
            assertTrue(encodedBytes < rawBytes / 2,
                    "expected TurboQuant payload to be much smaller: encoded=" + encodedBytes + " raw=" + rawBytes);
        }
    }

    @Test
    public void fixedBlocksPolicyHandlesSmallExactAndMultiBlockPrompts() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withBlockSize(4)
                .withPrefixCheckpointPolicy(KvBufferCacheSettings.PrefixCheckpointPolicy.FIXED_BLOCKS);
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);

        assertEquals(List.of(), cache.checkpointLengths(3));
        assertEquals(List.of(4), cache.checkpointLengths(4));
        assertEquals(List.of(4, 8, 12), cache.checkpointLengths(15));
    }

    @Test
    public void anchorsAndLargestPolicyKeepsSmallAnchorsAndLargestAlignedPrefix() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withBlockSize(4)
                .withMaxPrefixTokensPerPrompt(1000)
                .withMaxPrefixCheckpointsPerPrompt(4)
                .withPrefixCheckpointPolicy(KvBufferCacheSettings.PrefixCheckpointPolicy.ANCHORS_AND_LARGEST)
                .withPrefixCheckpointAnchors(List.of(4, 8, 12));
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);

        assertEquals(List.of(4, 8, 12, 32), cache.checkpointLengths(35));
    }

    @Test
    public void anchorsAndLargestPolicyHandlesSmallExactAndOverlappingPrompts() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withBlockSize(4)
                .withMaxPrefixCheckpointsPerPrompt(4)
                .withPrefixCheckpointPolicy(KvBufferCacheSettings.PrefixCheckpointPolicy.ANCHORS_AND_LARGEST)
                .withPrefixCheckpointAnchors(List.of(4, 8, 12));
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);

        assertEquals(List.of(), cache.checkpointLengths(3));
        assertEquals(List.of(4), cache.checkpointLengths(4));
        assertEquals(List.of(4, 8, 12), cache.checkpointLengths(15));
    }

    @Test
    public void anchorsAndLargestPolicyKeepsLargestWhenAnchorsExceedLimit() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withBlockSize(4)
                .withMaxPrefixCheckpointsPerPrompt(3)
                .withPrefixCheckpointPolicy(KvBufferCacheSettings.PrefixCheckpointPolicy.ANCHORS_AND_LARGEST)
                .withPrefixCheckpointAnchors(List.of(4, 8, 12, 16));
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);

        assertEquals(List.of(4, 8, 32), cache.checkpointLengths(35));
    }

    @Test
    public void anchorsAndLargestPolicyKeepsAnchorsAndCappedEndForLongPrompts() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withBlockSize(32)
                .withMaxPrefixTokensPerPrompt(512)
                .withMaxPrefixCheckpointsPerPrompt(4)
                .withPrefixCheckpointPolicy(KvBufferCacheSettings.PrefixCheckpointPolicy.ANCHORS_AND_LARGEST)
                .withPrefixCheckpointAnchors(List.of(32, 64, 128));
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);
        int cappedTokenLength = Math.min(1000, settings.getMaxPrefixTokensPerPrompt());

        assertEquals(List.of(32, 64, 128, 512), cache.checkpointLengths(cappedTokenLength));
    }

    @Test
    public void startAndEndPolicyKeepsHalfFromStartAndHalfFromEnd() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withBlockSize(4)
                .withMaxPrefixCheckpointsPerPrompt(6);
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);

        assertEquals(List.of(4, 8, 12, 24, 28, 32), cache.checkpointLengths(35));
    }

    @Test
    public void startAndEndPolicyHandlesSmallAndExactPrompts() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withBlockSize(4)
                .withMaxPrefixCheckpointsPerPrompt(6);
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);

        assertEquals(List.of(), cache.checkpointLengths(3));
        assertEquals(List.of(4), cache.checkpointLengths(4));
    }

    @Test
    public void startAndEndPolicyDeduplicatesOverlappingShortPrompts() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withBlockSize(4)
                .withMaxPrefixCheckpointsPerPrompt(6);
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);

        assertEquals(List.of(4, 8, 12), cache.checkpointLengths(15));
    }

    @Test
    public void startAndEndPolicyKeepsStartAndCappedEndForLongPrompts() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withBlockSize(32)
                .withMaxPrefixTokensPerPrompt(512)
                .withMaxPrefixCheckpointsPerPrompt(4);
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);
        int cappedTokenLength = Math.min(1000, settings.getMaxPrefixTokensPerPrompt());

        assertEquals(List.of(32, 64, 480, 512), cache.checkpointLengths(cappedTokenLength));
    }

    @Test
    public void startAndEndPolicyDeduplicatesWhenStartAndEndWindowsOverlap() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withBlockSize(32)
                .withMaxPrefixCheckpointsPerPrompt(4);
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);

        assertEquals(List.of(32, 64, 96), cache.checkpointLengths(100));
    }

    @Test
    public void defaultPolicyStoresStartAndEndCheckpoints() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withMaxEntries(512)
                .withBlockSize(4)
                .withMaxPrefixCheckpointsPerPrompt(4);
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);
        int[] tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        KvBufferCache.KvBuffer buf = cache.getEphemeralKvBuffer();

        cache.storePrefix(tokens, buf, Optional.empty());

        assertEquals(2, cache.prefixCache.size());
        assertEquals(8, cache.lookupPrefix(tokens, Optional.empty()).length());
    }

    @Test
    public void lookupUsesSameMaxPrefixCapAsStore() {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withMaxEntries(512)
                .withBlockSize(4)
                .withMaxPrefixTokensPerPrompt(16)
                .withPrefixCheckpointPolicy(KvBufferCacheSettings.PrefixCheckpointPolicy.ANCHORS_AND_LARGEST)
                .withPrefixCheckpointAnchors(List.of(4, 8));
        KvBufferCache cache = new KvBufferCache(mockModel(), settings);
        int[] tokens = new int[40];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = i + 1;
        }
        KvBufferCache.KvBuffer buf = cache.getEphemeralKvBuffer();

        cache.storePrefix(tokens, buf, Optional.empty());

        assertEquals(List.of(4, 8, 16), cache.checkpointLengths(16));
        assertEquals(16, cache.lookupPrefix(tokens, Optional.empty()).length());
    }

    private static int fillVisibleRows(AbstractTensor packed, AbstractTensor[] pages, int position, int windowStart, int rowWidth) {
        int packedRow = 0;
        int globalOffset = 0;
        for (AbstractTensor page : pages) {
            int pageRows = Math.min(page.shape().first(), (position + 1) - globalOffset);
            int overlapStart = Math.max(windowStart, globalOffset);
            int overlapEnd = Math.min(position + 1, globalOffset + pageRows);
            if (overlapStart < overlapEnd) {
                int rowOffset = overlapStart - globalOffset;
                int size = overlapEnd - overlapStart;
                for (int row = 0; row < size; row++) {
                    for (int col = 0; col < rowWidth; col++) {
                        packed.set(page.get(rowOffset + row, col), packedRow + row, col);
                    }
                }
                packedRow += size;
            }
            globalOffset += page.shape().first();
        }
        return packedRow;
    }

    private static void fillKv(KvBufferCache.KvBuffer buffer, Config config, int length) {
        for (int layer = 0; layer < config.numberOfLayers; layer++) {
            for (int pos = 0; pos < length; pos++) {
                try (AbstractTensor key = buffer.getKeyTensorForPosition(layer, pos);
                     AbstractTensor value = buffer.getValTensorForPosition(layer, pos)) {
                    for (int i = 0; i < config.kvLength; i++) {
                        key.set(kvValue(layer, pos, i, 1_000), 0, i);
                        value.set(kvValue(layer, pos, i, 2_000), 0, i);
                    }
                }
            }
        }
    }

    private static void assertKvPrefixEquals(KvBufferCache.KvBuffer expected, KvBufferCache.KvBuffer actual,
            Config config, int length) {
        for (int layer = 0; layer < config.numberOfLayers; layer++) {
            for (int pos = 0; pos < length; pos++) {
                try (AbstractTensor expectedKey = expected.getKeyTensorForPosition(layer, pos);
                     AbstractTensor actualKey = actual.getKeyTensorForPosition(layer, pos);
                     AbstractTensor expectedValue = expected.getValTensorForPosition(layer, pos);
                     AbstractTensor actualValue = actual.getValTensorForPosition(layer, pos)) {
                    for (int i = 0; i < config.kvLength; i++) {
                        assertEquals(expectedKey.get(0, i), actualKey.get(0, i), 0.0f,
                                "key layer=" + layer + " pos=" + pos + " index=" + i);
                        assertEquals(expectedValue.get(0, i), actualValue.get(0, i), 0.0f,
                                "value layer=" + layer + " pos=" + pos + " index=" + i);
                    }
                }
            }
        }
    }

    private static void assertKvPrefixClose(KvBufferCache.KvBuffer expected, KvBufferCache.KvBuffer actual,
            Config config, int length, float tolerance) {
        for (int layer = 0; layer < config.numberOfLayers; layer++) {
            for (int pos = 0; pos < length; pos++) {
                try (AbstractTensor expectedKey = expected.getKeyTensorForPosition(layer, pos);
                     AbstractTensor actualKey = actual.getKeyTensorForPosition(layer, pos);
                     AbstractTensor expectedValue = expected.getValTensorForPosition(layer, pos);
                     AbstractTensor actualValue = actual.getValTensorForPosition(layer, pos)) {
                    for (int i = 0; i < config.kvLength; i++) {
                        assertEquals(expectedKey.get(0, i), actualKey.get(0, i), tolerance,
                                "key layer=" + layer + " pos=" + pos + " index=" + i);
                        assertEquals(expectedValue.get(0, i), actualValue.get(0, i), tolerance,
                                "value layer=" + layer + " pos=" + pos + " index=" + i);
                    }
                }
            }
        }
    }

    private static void assertKvPrefixRmseBelow(KvBufferCache.KvBuffer expected, KvBufferCache.KvBuffer actual,
            Config config, int length, float maxRmse) {
        double squaredError = 0.0;
        long count = 0;
        for (int layer = 0; layer < config.numberOfLayers; layer++) {
            for (int pos = 0; pos < length; pos++) {
                try (AbstractTensor expectedKey = expected.getKeyTensorForPosition(layer, pos);
                     AbstractTensor actualKey = actual.getKeyTensorForPosition(layer, pos);
                     AbstractTensor expectedValue = expected.getValTensorForPosition(layer, pos);
                     AbstractTensor actualValue = actual.getValTensorForPosition(layer, pos)) {
                    for (int i = 0; i < config.kvLength; i++) {
                        double keyDiff = expectedKey.get(0, i) - actualKey.get(0, i);
                        double valueDiff = expectedValue.get(0, i) - actualValue.get(0, i);
                        squaredError += keyDiff * keyDiff + valueDiff * valueDiff;
                        count += 2;
                    }
                }
            }
        }
        double rmse = Math.sqrt(squaredError / count);
        assertTrue(rmse < maxRmse, "rmse=" + rmse + " max=" + maxRmse);
    }

    private static float kvValue(int layer, int pos, int index, int offset) {
        return offset + layer * 100 + pos * 10 + index;
    }

}
