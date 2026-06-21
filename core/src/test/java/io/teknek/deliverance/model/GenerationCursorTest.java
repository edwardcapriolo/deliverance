package io.teknek.deliverance.model;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GenerationCursorTest {

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
        int prefixLength = 8;

        GenerationCursor cursor = GenerationCursor.from(promptTokens, prefixLength);

        assertEquals(prefixLength, cursor.startPosition());
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
