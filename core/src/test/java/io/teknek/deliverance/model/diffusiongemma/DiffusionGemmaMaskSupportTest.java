package io.teknek.deliverance.model.diffusiongemma;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DiffusionGemmaMaskSupportTest {

    @Test
    void testDiffusionDecoderMaskNoCacheRaisesException() {
        assertThrows(IllegalArgumentException.class, () -> DiffusionGemmaMaskSupport.createDiffusionDecoderAttentionMask(
                tinyConfig(512), 2, 4, null, null));
    }

    @Test
    void testDiffusionDecoderMaskDynamicCache() {
        int prefillLength = 8;
        int canvasLength = 4;
        int batchSize = 2;
        try (FloatBufferTensor mask = ones(batchSize, prefillLength + canvasLength);
             DiffusionGemmaMaskSupport.DecoderAttentionMask mapping = DiffusionGemmaMaskSupport.createDiffusionDecoderAttentionMask(
                     tinyConfig(512), batchSize, canvasLength,
                     DiffusionGemmaMaskSupport.CacheState.dynamic(prefillLength), mask)) {
            assertNull(mapping.fullAttention());
            assertNull(mapping.slidingAttention());
        }

        try (DiffusionGemmaMaskSupport.DecoderAttentionMask mapping = DiffusionGemmaMaskSupport.createDiffusionDecoderAttentionMask(
                tinyConfig(512), batchSize, canvasLength,
                DiffusionGemmaMaskSupport.CacheState.dynamic(prefillLength), null)) {
            assertNull(mapping.fullAttention());
            assertNull(mapping.slidingAttention());
        }
    }

    @Test
    void testDiffusionDecoderMaskDynamicCacheLeftPadding() {
        int prefillLength = 8;
        int canvasLength = 4;
        int concatKvLength = prefillLength + canvasLength;
        int batchSize = 2;
        int leftPaddingLength = 2;
        int expectedNonZero = ((concatKvLength - leftPaddingLength) * canvasLength)
                + (concatKvLength * canvasLength);
        try (FloatBufferTensor mask = ones(batchSize, concatKvLength)) {
            for (int key = 0; key < leftPaddingLength; key++) {
                mask.set(0.0f, 0, key);
            }
            try (DiffusionGemmaMaskSupport.DecoderAttentionMask mapping = DiffusionGemmaMaskSupport.createDiffusionDecoderAttentionMask(
                    tinyConfig(512), batchSize, canvasLength,
                    DiffusionGemmaMaskSupport.CacheState.dynamic(prefillLength), mask)) {
                assertShape(mapping.fullAttention(), batchSize, 1, canvasLength, concatKvLength);
                assertEquals(expectedNonZero, sum(mapping.fullAttention()), 0.0f);
                assertEquals(0.0f, sum(mapping.fullAttention(), 0, leftPaddingLength), 0.0f);
                assertShape(mapping.slidingAttention(), batchSize, 1, canvasLength, concatKvLength);
                assertEquals(expectedNonZero, sum(mapping.slidingAttention()), 0.0f);
                assertEquals(0.0f, sum(mapping.slidingAttention(), 0, leftPaddingLength), 0.0f);
            }
        }
    }

    @Test
    void testDiffusionDecoderMaskDynamicCacheBeyondSlidingWindow() {
        int prefillLength = 16;
        int slidingWindowLength = 8;
        int canvasLength = 4;
        int concatKvLengthFull = prefillLength + canvasLength;
        int concatKvLengthSliding = slidingWindowLength + canvasLength - 1;
        int batchSize = 2;
        int leftPaddingLength = 2;
        int expectedNonZeroFull = ((concatKvLengthFull - leftPaddingLength) * canvasLength)
                + (concatKvLengthFull * canvasLength);
        int expectedNonZeroSliding = concatKvLengthSliding * batchSize * canvasLength;
        try (FloatBufferTensor mask = ones(batchSize, concatKvLengthFull)) {
            for (int key = 0; key < leftPaddingLength; key++) {
                mask.set(0.0f, 0, key);
            }
            try (DiffusionGemmaMaskSupport.DecoderAttentionMask mapping = DiffusionGemmaMaskSupport.createDiffusionDecoderAttentionMask(
                    tinyConfig(slidingWindowLength), batchSize, canvasLength,
                    DiffusionGemmaMaskSupport.CacheState.dynamic(prefillLength), mask)) {
                assertShape(mapping.fullAttention(), batchSize, 1, canvasLength, concatKvLengthFull);
                assertEquals(expectedNonZeroFull, sum(mapping.fullAttention()), 0.0f);
                assertEquals(0.0f, sum(mapping.fullAttention(), 0, leftPaddingLength), 0.0f);
                assertShape(mapping.slidingAttention(), batchSize, 1, canvasLength, concatKvLengthSliding);
                assertEquals(expectedNonZeroSliding, sum(mapping.slidingAttention()), 0.0f);
                assertEquals(leftPaddingLength * canvasLength,
                        sum(mapping.slidingAttention(), 0, leftPaddingLength), 0.0f);
            }
        }
    }

    @Test
    void testDiffusionDecoderMaskStaticCache() {
        int prefillLength = 8;
        int canvasLength = 4;
        int staticCacheLength = 16;
        int concatKvLength = staticCacheLength + canvasLength;
        int batchSize = 2;
        int expectedNonZero = (prefillLength + canvasLength) * canvasLength * batchSize;
        try (FloatBufferTensor mask = ones(batchSize, concatKvLength)) {
            zeroRange(mask, prefillLength, staticCacheLength);
            try (DiffusionGemmaMaskSupport.DecoderAttentionMask mapping = DiffusionGemmaMaskSupport.createDiffusionDecoderAttentionMask(
                    tinyConfig(512), batchSize, canvasLength,
                    DiffusionGemmaMaskSupport.CacheState.statik(prefillLength, staticCacheLength), mask)) {
                assertShape(mapping.fullAttention(), batchSize, 1, canvasLength, concatKvLength);
                assertEquals(expectedNonZero, sum(mapping.fullAttention()), 0.0f);
                assertShape(mapping.slidingAttention(), batchSize, 1, canvasLength, concatKvLength);
                assertEquals(expectedNonZero, sum(mapping.slidingAttention()), 0.0f);
            }
        }
    }

    @Test
    void testDiffusionDecoderMaskStaticCacheBadAttentionMask() {
        int prefillLength = 8;
        int canvasLength = 4;
        int staticCacheLength = 16;
        int batchSize = 2;
        try (FloatBufferTensor mask = ones(batchSize, staticCacheLength + canvasLength)) {
            assertThrows(IllegalArgumentException.class,
                    () -> DiffusionGemmaMaskSupport.createDiffusionDecoderAttentionMask(tinyConfig(512), batchSize,
                            canvasLength, DiffusionGemmaMaskSupport.CacheState.statik(prefillLength, staticCacheLength),
                            mask));
        }
    }

    @Test
    void testDiffusionDecoderMaskStaticCacheBeyondSlidingWindow() {
        int prefillLength = 16;
        int slidingWindowLength = 8;
        int canvasLength = 4;
        int staticCacheLength = 32;
        int concatKvLengthFull = staticCacheLength + canvasLength;
        int concatKvLengthSliding = slidingWindowLength + canvasLength;
        int batchSize = 2;
        int leftPaddingLength = 2;
        int expectedNonZeroFull = ((prefillLength + canvasLength - leftPaddingLength) * canvasLength)
                + ((prefillLength + canvasLength) * canvasLength);
        int expectedNonZeroSliding = concatKvLengthSliding * batchSize * canvasLength;
        try (FloatBufferTensor mask = ones(batchSize, concatKvLengthFull)) {
            for (int key = 0; key < leftPaddingLength; key++) {
                mask.set(0.0f, 0, key);
            }
            zeroRange(mask, prefillLength, staticCacheLength);
            try (DiffusionGemmaMaskSupport.DecoderAttentionMask mapping = DiffusionGemmaMaskSupport.createDiffusionDecoderAttentionMask(
                    tinyConfig(slidingWindowLength), batchSize, canvasLength,
                    DiffusionGemmaMaskSupport.CacheState.statik(prefillLength, staticCacheLength), mask)) {
                assertShape(mapping.fullAttention(), batchSize, 1, canvasLength, concatKvLengthFull);
                assertEquals(expectedNonZeroFull, sum(mapping.fullAttention()), 0.0f);
                assertEquals(0.0f, sum(mapping.fullAttention(), 0, leftPaddingLength), 0.0f);
                assertShape(mapping.slidingAttention(), batchSize, 1, canvasLength, concatKvLengthSliding);
                assertEquals(expectedNonZeroSliding, sum(mapping.slidingAttention()), 0.0f);
                assertEquals(leftPaddingLength * canvasLength,
                        sum(mapping.slidingAttention(), 0, leftPaddingLength), 0.0f);
            }
        }
    }

    private static DiffusionGemmaTextConfig tinyConfig(int slidingWindow) {
        return new DiffusionGemmaTextConfig(99, 32, 32, 2, 2, 2, 16, "gelu", 512, 0.02f,
                1.0e-6f, 0, 2, 1, true, null, false, 0.0f, slidingWindow,
                java.util.List.of("sliding_attention", "full_attention"), 30.0f,
                DiffusionGemmaTextConfig.BidirectionalAttention.VISION, 2, 16, 4, 2, 8);
    }

    private static FloatBufferTensor ones(int batchSize, int length) {
        FloatBufferTensor mask = new FloatBufferTensor(batchSize, length);
        for (int batch = 0; batch < batchSize; batch++) {
            for (int key = 0; key < length; key++) {
                mask.set(1.0f, batch, key);
            }
        }
        return mask;
    }

    private static void zeroRange(FloatBufferTensor mask, int startInclusive, int endExclusive) {
        for (int batch = 0; batch < mask.shape().first(); batch++) {
            for (int key = startInclusive; key < endExclusive; key++) {
                mask.set(0.0f, batch, key);
            }
        }
    }

    private static void assertShape(AbstractTensor tensor, int... shape) {
        for (int i = 0; i < shape.length; i++) {
            assertEquals(shape[i], tensor.shape().dim(i), "dim=" + i);
        }
    }

    private static float sum(AbstractTensor tensor) {
        float sum = 0.0f;
        for (int batch = 0; batch < tensor.shape().dim(0); batch++) {
            for (int head = 0; head < tensor.shape().dim(1); head++) {
                for (int query = 0; query < tensor.shape().dim(2); query++) {
                    for (int key = 0; key < tensor.shape().dim(3); key++) {
                        sum += tensor.get(batch, head, query, key);
                    }
                }
            }
        }
        return sum;
    }

    private static float sum(AbstractTensor tensor, int batch, int keyLimit) {
        float sum = 0.0f;
        for (int query = 0; query < tensor.shape().dim(2); query++) {
            for (int key = 0; key < keyLimit; key++) {
                sum += tensor.get(batch, 0, query, key);
            }
        }
        return sum;
    }
}
