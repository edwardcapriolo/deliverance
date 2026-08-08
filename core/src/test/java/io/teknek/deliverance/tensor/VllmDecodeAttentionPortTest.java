package io.teknek.deliverance.tensor;

import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Ports the core shapes from vLLM tests/kernels/attention/test_triton_decode_attention.py. */
class VllmDecodeAttentionPortTest {

    @ParameterizedTest(name = "B={0} L={1} H_Q={2} H_KV={3} D={4} page={5}")
    @CsvSource({
            "3,1027,32,32,128,1",
            "3,1027,32,32,128,16",
            "3,1025,32,8,128,1",
            "3,1025,32,8,128,16",
            "5,257,32,8,128,16"
    })
    void test_decode_attention(int batchSize, int seqLen, int numberOfHeads, int numberOfKeyValueHeads,
            int headSize, int pageSize) {
        int cacheSize = roundUp(seqLen + 32, pageSize);
        int maxPages = cacheSize / pageSize;
        int pagesPerBatch = roundUp(seqLen, pageSize) / pageSize;
        float scale = 1.0f / (float) Math.sqrt(headSize);

        try (AbstractTensor query = tensor(batchSize * numberOfHeads, headSize, 3);
             AbstractTensor flatKey = tensor(cacheSize * numberOfKeyValueHeads, headSize, 7);
             AbstractTensor flatValue = tensor(cacheSize * numberOfKeyValueHeads, headSize, 11);
             AbstractTensor pagedKey = new FloatBufferTensor(maxPages * pageSize * numberOfKeyValueHeads, headSize);
             AbstractTensor pagedValue = new FloatBufferTensor(maxPages * pageSize * numberOfKeyValueHeads, headSize);
             AbstractTensor flatOut = new FloatBufferTensor(batchSize * numberOfHeads, headSize);
             AbstractTensor pagedOut = new FloatBufferTensor(batchSize * numberOfHeads, headSize)) {

            int[][] blockTable = new int[batchSize][pagesPerBatch];
            int[][] tokenTable = new int[batchSize][seqLen];
            for (int batch = 0; batch < batchSize; batch++) {
                for (int logicalPage = 0; logicalPage < pagesPerBatch; logicalPage++) {
                    int physicalPage = (batch * pagesPerBatch + logicalPage * 7 + 3) % maxPages;
                    blockTable[batch][logicalPage] = physicalPage;
                    for (int row = 0; row < pageSize; row++) {
                        int logicalRow = logicalPage * pageSize + row;
                        if (logicalRow < seqLen) {
                            tokenTable[batch][logicalRow] = physicalPage * pageSize + row;
                        }
                    }
                }
            }

            packFlatCacheIntoPagedCache(flatKey, flatValue, pagedKey, pagedValue, numberOfKeyValueHeads, headSize,
                    tokenTable, batchSize, seqLen, pageSize, blockTable);

            decodeFlat(flatOut, query, flatKey, flatValue, tokenTable, batchSize, seqLen, numberOfHeads,
                    numberOfKeyValueHeads, headSize, scale);
            decodePaged(pagedOut, query, pagedKey, pagedValue, blockTable, batchSize, seqLen, pageSize, numberOfHeads,
                    numberOfKeyValueHeads, headSize, scale);

            assertTensorEquals(flatOut, pagedOut, 1.0e-5f);
        }
    }

    @Disabled("vLLM FP8 decode-attention path depends on torch float8_e4m3fn and KV scale kernels; Deliverance KV cache does not have fp8 storage yet.")
    @ParameterizedTest
    @CsvSource({ "3,1025,32,8,128,16" })
    void test_decode_attention_fp8(int batchSize, int seqLen, int numberOfHeads, int numberOfKeyValueHeads,
            int headSize, int pageSize) {
    }

    @Disabled("vLLM cross-layer view test validates inflated page-dim stride in torch views; Deliverance does not expose cross-layer GPU KV cache views yet.")
    @ParameterizedTest
    @CsvSource({ "32,8,128,128,16" })
    void test_decode_attention_cross_layer_view(int numberOfHeads, int numberOfKeyValueHeads, int headSize,
            int valueHeadSize, int pageSize) {
    }

    private static void packFlatCacheIntoPagedCache(AbstractTensor flatKey, AbstractTensor flatValue,
            AbstractTensor pagedKey, AbstractTensor pagedValue, int numberOfKeyValueHeads, int headSize,
            int[][] tokenTable, int batchSize, int seqLen, int pageSize, int[][] blockTable) {
        for (int batch = 0; batch < batchSize; batch++) {
            for (int logicalRow = 0; logicalRow < seqLen; logicalRow++) {
                int physicalSlot = tokenTable[batch][logicalRow];
                int logicalPage = logicalRow / pageSize;
                int blockOffset = logicalRow % pageSize;
                int physicalPage = blockTable[batch][logicalPage];
                assertEquals(physicalSlot, physicalPage * pageSize + blockOffset);
                for (int kvHead = 0; kvHead < numberOfKeyValueHeads; kvHead++) {
                    int srcRow = physicalSlot * numberOfKeyValueHeads + kvHead;
                    int dstRow = (physicalPage * pageSize + blockOffset) * numberOfKeyValueHeads + kvHead;
                    pagedKey.copyFrom(flatKey, flatKey.getOffset(srcRow, 0), pagedKey.getOffset(dstRow, 0), headSize);
                    pagedValue.copyFrom(flatValue, flatValue.getOffset(srcRow, 0), pagedValue.getOffset(dstRow, 0), headSize);
                }
            }
        }
    }

    private static void decodeFlat(AbstractTensor out, AbstractTensor query, AbstractTensor key, AbstractTensor value,
            int[][] tokenTable, int batchSize, int seqLen, int numberOfHeads, int numberOfKeyValueHeads, int headSize,
            float scale) {
        int headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        for (int batch = 0; batch < batchSize; batch++) {
            for (int head = 0; head < numberOfHeads; head++) {
                int kvHead = head / headGroupSize;
                int outRow = batch * numberOfHeads + head;
                decodeOne(out, outRow, query, outRow, key, value, tokenTable[batch], kvHead, numberOfKeyValueHeads,
                        headSize, scale);
            }
        }
    }

    private static void decodePaged(AbstractTensor out, AbstractTensor query, AbstractTensor key, AbstractTensor value,
            int[][] blockTable, int batchSize, int seqLen, int pageSize, int numberOfHeads, int numberOfKeyValueHeads,
            int headSize, float scale) {
        int headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        for (int batch = 0; batch < batchSize; batch++) {
            int[] slots = new int[seqLen];
            for (int logicalRow = 0; logicalRow < seqLen; logicalRow++) {
                int logicalPage = logicalRow / pageSize;
                int blockOffset = logicalRow % pageSize;
                slots[logicalRow] = blockTable[batch][logicalPage] * pageSize + blockOffset;
            }
            for (int head = 0; head < numberOfHeads; head++) {
                int kvHead = head / headGroupSize;
                int outRow = batch * numberOfHeads + head;
                decodeOne(out, outRow, query, outRow, key, value, slots, kvHead, numberOfKeyValueHeads, headSize,
                        scale);
            }
        }
    }

    private static void decodeOne(AbstractTensor out, int outRow, AbstractTensor query, int queryRow,
            AbstractTensor key, AbstractTensor value, int[] slots, int kvHead, int numberOfKeyValueHeads, int headSize,
            float scale) {
        for (int col = 0; col < headSize; col++) {
            out.set(0.0f, outRow, col);
        }
        float max = Float.NEGATIVE_INFINITY;
        float denom = 0.0f;
        for (int slot : slots) {
            int kvRow = slot * numberOfKeyValueHeads + kvHead;
            float score = 0.0f;
            for (int col = 0; col < headSize; col++) {
                score += query.get(queryRow, col) * key.get(kvRow, col);
            }
            score *= scale;
            float nextMax = Math.max(max, score);
            float oldScale = max == Float.NEGATIVE_INFINITY ? 0.0f : (float) Math.exp(max - nextMax);
            float weight = (float) Math.exp(score - nextMax);
            for (int col = 0; col < headSize; col++) {
                out.set(out.get(outRow, col) * oldScale + weight * value.get(kvRow, col), outRow, col);
            }
            denom = denom * oldScale + weight;
            max = nextMax;
        }
        for (int col = 0; col < headSize; col++) {
            out.set(out.get(outRow, col) / denom, outRow, col);
        }
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

    private static void assertTensorEquals(AbstractTensor expected, AbstractTensor actual, float tolerance) {
        assertEquals(expected.shape(), actual.shape());
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int col = 0; col < expected.shape().last(); col++) {
                assertEquals(expected.get(row, col), actual.get(row, col), tolerance,
                        "row=" + row + " col=" + col);
            }
        }
    }

    private static int roundUp(int value, int multiple) {
        return ((value + multiple - 1) / multiple) * multiple;
    }
}
