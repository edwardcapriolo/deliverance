package io.teknek.deliverance.model;

import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GenerationCursorTest {

    private static final int KV_BLOCK_SIZE = new KvBufferCacheSettings(true).getBlockSize();

    @Test
    void noPrefixHitProcessesWholePromptAndDecodesAfterPrompt() {
        int[] promptTokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        GenerationCursor cursor = GenerationCursor.from(promptTokens, 0);

        assertEquals(0, cursor.startPosition());
        assertArrayEquals(promptTokens, cursor.tokensToProcess());
        assertEquals(promptTokens.length, cursor.decodeStartPosition());
    }

    @Test
    void partialPrefixHitProcessesOnlyUncachedPromptAndDecodesAfterPrompt() {
        int[] promptTokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        GenerationCursor cursor = GenerationCursor.from(promptTokens, KV_BLOCK_SIZE);

        assertEquals(KV_BLOCK_SIZE, cursor.startPosition());
        assertArrayEquals(new int[]{9, 10}, cursor.tokensToProcess());
        assertEquals(promptTokens.length, cursor.decodeStartPosition());
    }

    @Test
    void fullPrefixHitReplaysLastPromptTokenAtLastPromptPositionAndDecodesAfterPrompt() {
        int[] promptTokens = {1, 2, 3, 4, 5, 6, 7, 8};

        GenerationCursor cursor = GenerationCursor.from(promptTokens, promptTokens.length);

        assertEquals(promptTokens.length, cursor.startPosition());
        assertArrayEquals(new int[0], cursor.tokensToProcess());
        assertEquals(8, cursor.replayToken());
        assertEquals(promptTokens.length - 1, cursor.replayPosition());
        assertEquals(promptTokens.length, cursor.decodeStartPosition());
    }
}
