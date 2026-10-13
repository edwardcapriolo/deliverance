package io.teknek.deliverance.model.embeddinggemma2;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Direct contract tests for EmbeddingGemma 2 full and sliding attention visibility. */
class EmbeddingGemma2AttentionMaskTest {
    @Test
    void fullAttentionIsBidirectionalAndUnbounded() {
        int[] valid = {1, 1, 1, 1, 1};
        for (int query = 0; query < valid.length; query++) {
            for (int key = 0; key < valid.length; key++) {
                assertTrue(EmbeddingGemma2AttentionMask.allows(
                        "full_attention", query, key, 1, valid));
            }
        }
    }

    @Test
    void slidingAttentionUsesSymmetricInclusiveRadius() {
        int[] valid = {1, 1, 1, 1, 1};
        int radius = 1;
        for (int query = 0; query < valid.length; query++) {
            for (int key = 0; key < valid.length; key++) {
                boolean expected = Math.abs(query - key) <= radius;
                assertTrue(expected == EmbeddingGemma2AttentionMask.allows(
                        "sliding_attention", query, key, radius, valid),
                        "query=" + query + " key=" + key);
            }
        }
        assertTrue(EmbeddingGemma2AttentionMask.allows("sliding_attention", 0, 1, radius, valid));
        assertTrue(EmbeddingGemma2AttentionMask.allows("sliding_attention", 1, 0, radius, valid));
        assertFalse(EmbeddingGemma2AttentionMask.allows("sliding_attention", 0, 2, radius, valid));
    }

    @Test
    void paddingMasksKeysForBothAttentionTypes() {
        int[] mask = {1, 1, 0, 1};
        assertFalse(EmbeddingGemma2AttentionMask.allows("full_attention", 0, 2, 10, mask));
        assertFalse(EmbeddingGemma2AttentionMask.allows("sliding_attention", 3, 2, 10, mask));
        assertTrue(EmbeddingGemma2AttentionMask.allows("full_attention", 0, 3, 0, mask));
    }

    @Test
    void rejectsInvalidLayerAndPositions() {
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2AttentionMask.allows("causal_attention", 0, 0, 1, null));
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2AttentionMask.allows("full_attention", -1, 0, 1, null));
        assertThrows(IllegalArgumentException.class,
                () -> EmbeddingGemma2AttentionMask.allows("sliding_attention", 0, 0, -1, null));
    }
}
