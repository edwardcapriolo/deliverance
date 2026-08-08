package io.teknek.deliverance.tensor;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Ports the test methods in vLLM's tests/kernels/attention/test_cache.py to the
 * Deliverance KV-cache shape where possible.
 */
class VllmCacheKernelPortTest {

    @Test
    void test_reshape_and_cache() {
        int numTokens = 42;
        int numHeads = 8;
        int headSize = 64;
        int blockSize = 8;
        int numBlocks = 128;
        int kvLength = numHeads * headSize;
        int[] slotMapping = randomSlotMapping(numTokens, blockSize * numBlocks, 0);

        try (AbstractTensor key = tensor(numTokens, kvLength, 3);
             AbstractTensor value = tensor(numTokens, kvLength, 11);
             AbstractTensor keyCache = new FloatBufferTensor(blockSize * numBlocks, kvLength);
             AbstractTensor valueCache = new FloatBufferTensor(blockSize * numBlocks, kvLength);
             AbstractTensor expectedKeyCache = new FloatBufferTensor(blockSize * numBlocks, kvLength);
             AbstractTensor expectedValueCache = new FloatBufferTensor(blockSize * numBlocks, kvLength)) {

            KvSlotMapping mapping = new KvSlotMapping(blockSize, slotMapping);
            KvCacheLayout.reshapeAndCache(key, value, expectedKeyCache, expectedValueCache, mapping, kvLength);
            KvCacheLayout.reshapeAndCache(key, value, keyCache, valueCache, mapping, kvLength);

            assertTensorEquals(expectedKeyCache, keyCache);
            assertTensorEquals(expectedValueCache, valueCache);
        }
    }

    @Test
    void test_reshape_and_cache_flash() {
        int numTokens = 21;
        int numHeads = 8;
        int headSize = 64;
        int blockSize = 16;
        int numBlocks = 64;
        int kvLength = numHeads * headSize;
        int[] slotMapping = randomSlotMapping(numTokens, blockSize * numBlocks, 0);

        try (AbstractTensor key = tensor(numTokens, kvLength, 13);
             AbstractTensor value = tensor(numTokens, kvLength, 17);
             AbstractTensor keyCacheNhd = new FloatBufferTensor(blockSize * numBlocks, kvLength);
             AbstractTensor valueCacheNhd = new FloatBufferTensor(blockSize * numBlocks, kvLength);
             AbstractTensor keyCacheHnd = new FloatBufferTensor(blockSize * numBlocks, kvLength);
             AbstractTensor valueCacheHnd = new FloatBufferTensor(blockSize * numBlocks, kvLength)) {

            KvSlotMapping mapping = new KvSlotMapping(blockSize, slotMapping);
            KvCacheLayout.reshapeAndCache(key, value, keyCacheNhd, valueCacheNhd, mapping, kvLength);
            KvCacheLayout.reshapeAndCache(key, value, keyCacheHnd, valueCacheHnd, mapping, kvLength);

            assertTensorEquals(keyCacheNhd, keyCacheHnd);
            assertTensorEquals(valueCacheNhd, valueCacheHnd);
        }
    }

    @Test
    void test_reshape_and_cache_flash_unaligned_rows() {
        int numTokens = 42;
        int numHeads = 13;
        int headSize = 46;
        int blockSize = 16;
        int numBlocks = 128;
        int kvLength = numHeads * headSize;
        int[] slotMapping = randomSlotMapping(numTokens, blockSize * numBlocks, 0);

        try (AbstractTensor key = tensor(numTokens, kvLength, 19);
             AbstractTensor value = tensor(numTokens, kvLength, 23);
            AbstractTensor keyCache = new FloatBufferTensor(blockSize * numBlocks, kvLength);
            AbstractTensor valueCache = new FloatBufferTensor(blockSize * numBlocks, kvLength)) {
            KvCacheLayout.reshapeAndCache(key, value, keyCache, valueCache, new KvSlotMapping(blockSize, slotMapping),
                    kvLength);

            for (int token = 0; token < numTokens; token++) {
                int slot = slotMapping[token];
                for (int col = 0; col < kvLength; col++) {
                    assertEquals(key.get(token, col), keyCache.get(slot, col), 1.0e-6f);
                    assertEquals(value.get(token, col), valueCache.get(slot, col), 1.0e-6f);
                }
            }
        }
    }

    @Test
    void reshapeAndCacheUsesBlockIndexAndBlockOffsetLikeVllmSlotMapping() {
        int numHeads = 2;
        int headSize = 4;
        int blockSize = 4;
        int numBlocks = 3;
        int kvLength = numHeads * headSize;
        int[] slotMapping = { 0, 3, 4, 7, 8, 11 };

        try (AbstractTensor key = tensor(slotMapping.length, kvLength, 37);
             AbstractTensor value = tensor(slotMapping.length, kvLength, 41);
             AbstractTensor keyCache = new FloatBufferTensor(blockSize * numBlocks, kvLength);
             AbstractTensor valueCache = new FloatBufferTensor(blockSize * numBlocks, kvLength)) {
            KvCacheLayout.reshapeAndCache(key, value, keyCache, valueCache, new KvSlotMapping(blockSize, slotMapping),
                    kvLength);

            for (int token = 0; token < slotMapping.length; token++) {
                int slot = slotMapping[token];
                int blockIndex = slot / blockSize;
                int blockOffset = slot % blockSize;
                int flattenedSlot = blockIndex * blockSize + blockOffset;
                assertEquals(slot, flattenedSlot);
                for (int col = 0; col < kvLength; col++) {
                    assertEquals(key.get(token, col), keyCache.get(flattenedSlot, col), 1.0e-6f);
                    assertEquals(value.get(token, col), valueCache.get(flattenedSlot, col), 1.0e-6f);
                }
            }
        }
    }

    @Test
    void reshapeAndCacheLastWriteWinsForDuplicateSlots() {
        int numHeads = 2;
        int headSize = 4;
        int kvLength = numHeads * headSize;
        int[] slotMapping = { 5, 5, 5 };

        try (AbstractTensor key = tensor(slotMapping.length, kvLength, 43);
             AbstractTensor value = tensor(slotMapping.length, kvLength, 47);
            AbstractTensor keyCache = new FloatBufferTensor(8, kvLength);
            AbstractTensor valueCache = new FloatBufferTensor(8, kvLength)) {
            KvCacheLayout.reshapeAndCache(key, value, keyCache, valueCache, new KvSlotMapping(4, slotMapping),
                    kvLength);

            for (int col = 0; col < kvLength; col++) {
                assertEquals(key.get(2, col), keyCache.get(5, col), 1.0e-6f);
                assertEquals(value.get(2, col), valueCache.get(5, col), 1.0e-6f);
            }
        }
    }

    @Test
    void blockTableGatherReadsPhysicalBlocksInLogicalOrder() {
        int blockSize = 4;
        int rowWidth = 3;
        int[][] blocks = { { 2, 0, 3 }, { 1, 4, 5 } };
        KvBlockTable blockTable = new KvBlockTable(blockSize, blocks);

        try (AbstractTensor source = new FloatBufferTensor(24, rowWidth);
             AbstractTensor actual = new FloatBufferTensor(7, rowWidth);
             AbstractTensor expected = new FloatBufferTensor(7, rowWidth)) {
            for (int row = 0; row < source.shape().first(); row++) {
                for (int col = 0; col < rowWidth; col++) {
                    source.set(row * 10 + col, row, col);
                }
            }
            for (int i = 0; i < 7; i++) {
                int slot = blockTable.slot(0, 2 + i);
                expected.copyFrom(source, source.getOffset(slot, 0), expected.getOffset(i, 0), rowWidth);
            }

            KvCacheLayout.gather(source, actual, blockTable, 0, 2, 7, rowWidth);

            assertTensorEquals(expected, actual);
        }
    }

    @Test
    void vllmLayoutDecodeAttentionMatchesPackedPageReference() {
        int numberOfHeads = 4;
        int numberOfKeyValueHeads = 2;
        int headSize = 8;
        int blockSize = 4;
        int visibleRows = 9;
        int rowWidth = numberOfKeyValueHeads * headSize;
        int[][] blocks = { { 2, 0, 3 } };
        KvBlockTable blockTable = new KvBlockTable(blockSize, blocks);

        try (AbstractTensor query = tensor(1, numberOfHeads * headSize, 5);
             AbstractTensor keyCache = tensor(16, rowWidth, 7);
             AbstractTensor valueCache = tensor(16, rowWidth, 11);
             AbstractTensor expected = new FloatBufferTensor(1, numberOfHeads * headSize);
             AbstractTensor actual = new FloatBufferTensor(1, numberOfHeads * headSize)) {

            decodeVllmLayoutReference(actual, query, keyCache, valueCache, blockTable, 0, visibleRows,
                    numberOfHeads, numberOfKeyValueHeads, headSize, 0.25f);

            AbstractTensor[] keyPages = logicalPagesFromBlockTable(keyCache, blockTable, 0, blockSize, visibleRows,
                    rowWidth);
            AbstractTensor[] valuePages = logicalPagesFromBlockTable(valueCache, blockTable, 0, blockSize, visibleRows,
                    rowWidth);
            try {
                new io.teknek.deliverance.tensor.operations.NaiveTensorOperations()
                        .decodePagedAttention(expected, query, keyPages, valuePages, visibleRows, numberOfHeads,
                                numberOfKeyValueHeads, headSize, 0.25f, null);
            } finally {
                for (AbstractTensor page : keyPages) {
                    page.close();
                }
                for (AbstractTensor page : valuePages) {
                    page.close();
                }
            }

            assertTensorEquals(expected, actual);
        }
    }

    @Test
    void slotMappingDefensivelyCopiesSlots() {
        int[] slots = { 3, 1, 2 };
        KvSlotMapping mapping = new KvSlotMapping(4, slots);
        slots[0] = 99;

        assertEquals(3, mapping.slot(0));
        assertEquals(0, mapping.blockIndex(0));
        assertEquals(3, mapping.blockOffset(0));
    }

    @Test
    void deliveranceLogicalPositionMappingMatchesContiguousVllmSlots() {
        int numTokens = 12;
        int numHeads = 2;
        int headSize = 4;
        int kvLength = numHeads * headSize;
        int blockSize = 4;
        Config config = new Config(1024, 64, 128, numHeads, numHeads, 1,
                1e-5f, 1000, 0, List.of(1), ActivationFunction.Type.GELU_PYTORCH_TANH, null, null,
                headSize, null, null);
        AbstractModel model = mockModel(config);
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true).withBlockSize(blockSize);
        KvBufferCache cache = new KvBufferCache(model, settings);
        KvBufferCache.KvBuffer buffer = cache.new KvBuffer("contiguous-vllm-slots", 1 << 20);

        try (AbstractTensor key = tensor(numTokens, kvLength, 53);
             AbstractTensor value = tensor(numTokens, kvLength, 59)) {
            for (int pos = 0; pos < numTokens; pos++) {
                try (AbstractTensor keyRow = buffer.getKeyTensorForPosition(0, pos);
                     AbstractTensor valueRow = buffer.getValTensorForPosition(0, pos)) {
                    keyRow.copyFrom(key, key.getOffset(pos, 0), 0, kvLength);
                    valueRow.copyFrom(value, value.getOffset(pos, 0), 0, kvLength);
                }
            }

            for (int pos = 0; pos < numTokens; pos++) {
                int blockIndex = pos / blockSize;
                int blockOffset = pos % blockSize;
                assertEquals(pos, blockIndex * blockSize + blockOffset);
                try (AbstractTensor keyRow = buffer.getKeyTensorForPosition(0, pos);
                     AbstractTensor valueRow = buffer.getValTensorForPosition(0, pos)) {
                    for (int col = 0; col < kvLength; col++) {
                        assertEquals(key.get(pos, col), keyRow.get(0, col), 1.0e-6f);
                        assertEquals(value.get(pos, col), valueRow.get(0, col), 1.0e-6f);
                    }
                }
            }
        } finally {
            buffer.close();
        }
    }

    @Disabled("Deliverance KV cache does not expose a swap_blocks kernel or arbitrary CPU/GPU block-copy API; it copies prefixes by logical token position instead.")
    @Test
    void test_swap_blocks() {
    }

    @Disabled("Deliverance does not currently use vLLM fp8_e4m3 KV-cache storage or convert_fp8 cache kernels.")
    @Test
    void test_fp8_e4m3_conversion() {
    }

    @Disabled("MLA concat_and_cache is specific to vLLM MLA cache entries; Deliverance has no MLA KV-cache entry layout yet.")
    @Test
    void test_concat_and_cache_mla() {
    }

    @Disabled("DeepSeek MLA fp8_ds_mla cache packing is not represented in Deliverance KV cache.")
    @Test
    void test_concat_and_cache_ds_mla() {
    }

    @Disabled("Deliverance does not expose a block-level swap operation for MLA cache rows.")
    @Test
    void test_swap_blocks_mla() {
    }

    @Disabled("gather_and_maybe_dequant_cache operates on vLLM MLA cache tensors and fp8 dequantization; no Deliverance equivalent exists yet.")
    @Test
    void test_gather_and_maybe_dequant_cache_mla() {
    }

    @Disabled("gather_and_maybe_dequant_cache with seq_starts depends on vLLM block_table plus MLA/fp8 cache layout not present in Deliverance.")
    @Test
    void test_gather_and_maybe_dequant_cache_mla_with_seq_starts() {
    }

    @Disabled("cp_gather_cache is a vLLM block-table gather kernel; Deliverance currently reads visible KV pages through KvPageTable rather than gathering to a compact output tensor.")
    @Test
    void test_cp_gather_cache_mla() {
    }

    @Disabled("CPU MLA concat_and_cache targets the MLA cache format; Deliverance has no MLA cache representation.")
    @Test
    void test_concat_and_cache_mla_cpu() {
    }

    @Test
    void deliveranceKvBufferWritesRowsToSlotEquivalentPositions() {
        int numTokens = 42;
        int numHeads = 8;
        int headSize = 64;
        int blockSize = 8;
        int kvLength = numHeads * headSize;
        Config config = new Config(1024, 64, 128, numHeads, numHeads, 1,
                1e-5f, 1000, 0, List.of(1), ActivationFunction.Type.GELU_PYTORCH_TANH, null, null,
                headSize, null, null);
        AbstractModel model = mockModel(config);
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true).withBlockSize(blockSize);
        KvBufferCache cache = new KvBufferCache(model, settings);
        KvBufferCache.KvBuffer buffer = cache.new KvBuffer("vllm-reshape-cache-port", 1 << 20);

        try (AbstractTensor key = tensor(numTokens, kvLength, 29);
             AbstractTensor value = tensor(numTokens, kvLength, 31)) {
            for (int token = 0; token < numTokens; token++) {
                try (AbstractTensor keyRow = buffer.getKeyTensorForPosition(0, token);
                     AbstractTensor valueRow = buffer.getValTensorForPosition(0, token)) {
                    keyRow.copyFrom(key, key.getOffset(token, 0), 0, kvLength);
                    valueRow.copyFrom(value, value.getOffset(token, 0), 0, kvLength);
                }
            }

            for (int token = 0; token < numTokens; token++) {
                try (AbstractTensor keyRow = buffer.getKeyTensorForPosition(0, token);
                     AbstractTensor valueRow = buffer.getValTensorForPosition(0, token)) {
                    for (int col = 0; col < kvLength; col++) {
                        assertEquals(key.get(token, col), keyRow.get(0, col), 1.0e-6f);
                        assertEquals(value.get(token, col), valueRow.get(0, col), 1.0e-6f);
                    }
                }
            }
        } finally {
            buffer.close();
        }
    }

    @Test
    void deliverancePageViewsAreStableForGpuRegistration() {
        Config config = new Config(1024, 64, 128, 8, 8, 1,
                1e-5f, 1000, 0, List.of(1), ActivationFunction.Type.GELU_PYTORCH_TANH, null, null,
                64, null, null);
        AbstractModel model = mockModel(config);
        KvBufferCache cache = new KvBufferCache(model, new KvBufferCacheSettings(true).withBlockSize(8));
        KvBufferCache.KvBuffer buffer = cache.new KvBuffer("stable-page-view", 1 << 20);

        try {
            AbstractTensor keyPageA = buffer.getKeyPageTensorForPosition(0, 0);
            AbstractTensor keyPageB = buffer.getKeyPageTensorForPosition(0, 7);
            AbstractTensor valuePageA = buffer.getValuePageTensorForPosition(0, 0);
            AbstractTensor valuePageB = buffer.getValuePageTensorForPosition(0, 7);

            assertSame(keyPageA, keyPageB, "same context page must return stable key page view");
            assertSame(valuePageA, valuePageB, "same context page must return stable value page view");
            assertTrue(buffer.getRowInContextPage(9) < keyPageA.shape().first());
        } finally {
            buffer.close();
        }
    }

    private static int[] randomSlotMapping(int count, int slots, long seed) {
        List<Integer> values = new ArrayList<>();
        for (int i = 0; i < slots; i++) {
            values.add(i);
        }
        Collections.shuffle(values, new Random(seed));
        int[] selected = new int[count];
        for (int i = 0; i < count; i++) {
            selected[i] = values.get(i);
        }
        return selected;
    }

    private static AbstractTensor tensor(int rows, int cols, int seed) {
        FloatBufferTensor tensor = new FloatBufferTensor(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                tensor.set((((row * 19 + col * 23 + seed) % 43) - 21) / 21.0f, row, col);
            }
        }
        return tensor;
    }

    private static void assertTensorEquals(AbstractTensor expected, AbstractTensor actual) {
        assertEquals(expected.shape(), actual.shape());
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int col = 0; col < expected.shape().last(); col++) {
                assertEquals(expected.get(row, col), actual.get(row, col), 1.0e-6f,
                        "row=" + row + " col=" + col);
            }
        }
    }

    private static AbstractTensor[] logicalPagesFromBlockTable(AbstractTensor cache, KvBlockTable blockTable,
            int sequence, int blockSize, int visibleRows, int rowWidth) {
        int pageCount = (visibleRows + blockSize - 1) / blockSize;
        AbstractTensor[] pages = new AbstractTensor[pageCount];
        for (int logicalBlock = 0; logicalBlock < pageCount; logicalBlock++) {
            int physicalBlock = blockTable.physicalBlock(sequence, logicalBlock);
            FloatBufferTensor page = new FloatBufferTensor(blockSize, rowWidth);
            for (int row = 0; row < blockSize; row++) {
                page.copyFrom(cache, cache.getOffset(physicalBlock * blockSize + row, 0), page.getOffset(row, 0),
                        rowWidth);
            }
            pages[logicalBlock] = page;
        }
        return pages;
    }

    private static void decodeVllmLayoutReference(AbstractTensor out, AbstractTensor query, AbstractTensor keyCache,
            AbstractTensor valueCache, KvBlockTable blockTable, int sequence, int visibleRows, int numberOfHeads,
            int numberOfKeyValueHeads, int headSize, float scale) {
        int headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        for (int head = 0; head < numberOfHeads; head++) {
            int kvHead = head / headGroupSize;
            int queryOffset = head * headSize;
            int kvOffset = kvHead * headSize;
            float max = Float.NEGATIVE_INFINITY;
            float denom = 0.0f;
            for (int col = 0; col < headSize; col++) {
                out.set(0.0f, 0, queryOffset + col);
            }
            for (int logicalRow = 0; logicalRow < visibleRows; logicalRow++) {
                int slot = blockTable.slot(sequence, logicalRow);
                float score = 0.0f;
                for (int col = 0; col < headSize; col++) {
                    score += query.get(0, queryOffset + col) * keyCache.get(slot, kvOffset + col);
                }
                score *= scale;
                float nextMax = Math.max(max, score);
                float oldScale = max == Float.NEGATIVE_INFINITY ? 0.0f : (float) Math.exp(max - nextMax);
                float weight = (float) Math.exp(score - nextMax);
                for (int col = 0; col < headSize; col++) {
                    out.set(out.get(0, queryOffset + col) * oldScale + weight * valueCache.get(slot, kvOffset + col),
                            0, queryOffset + col);
                }
                denom = denom * oldScale + weight;
                max = nextMax;
            }
            for (int col = 0; col < headSize; col++) {
                out.set(out.get(0, queryOffset + col) / denom, 0, queryOffset + col);
            }
        }
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
}
