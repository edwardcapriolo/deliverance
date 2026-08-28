package io.teknek.deliverance.model.diffusiongemma;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DiffusionGemmaGenerationSupportTest {

    @Test
    void testTokensPerForward() {
        int[][] inputIds = filled(1, 100, 5);
        int[] decoderForwardPasses = {10};
        int padTokenId = 1;

        assertArrayEquals(new float[] {10.0f}, DiffusionGemmaGenerationSupport.computeTokensPerForward(
                inputIds, decoderForwardPasses, 0, padTokenId));

        assertArrayEquals(new float[] {9.0f}, DiffusionGemmaGenerationSupport.computeTokensPerForward(
                inputIds, decoderForwardPasses, 10, padTokenId));

        fillTail(inputIds[0], 30, padTokenId);
        assertArrayEquals(new float[] {6.0f}, DiffusionGemmaGenerationSupport.computeTokensPerForward(
                inputIds, decoderForwardPasses, 10, padTokenId));
    }

    @Test
    void testTokensPerForwardBatched() {
        int[][] inputIds = filled(2, 100, 5);
        int[] decoderForwardPasses = {10, 7};
        int padTokenId = 1;

        assertArrayEquals(new float[] {10.0f, 100.0f / 7.0f},
                DiffusionGemmaGenerationSupport.computeTokensPerForward(inputIds, decoderForwardPasses, 0,
                        padTokenId), 1.0e-6f);

        assertArrayEquals(new float[] {9.0f, 90.0f / 7.0f},
                DiffusionGemmaGenerationSupport.computeTokensPerForward(inputIds, decoderForwardPasses, 10,
                        padTokenId), 1.0e-6f);

        fillTail(inputIds[0], 30, padTokenId);
        fillTail(inputIds[1], 15, padTokenId);
        assertArrayEquals(new float[] {6.0f, 75.0f / 7.0f},
                DiffusionGemmaGenerationSupport.computeTokensPerForward(inputIds, decoderForwardPasses, 10,
                        padTokenId), 1.0e-6f);
    }

    @Test
    void rejectsInvalidInputs() {
        assertThrows(IllegalArgumentException.class, () -> DiffusionGemmaGenerationSupport.computeTokensPerForward(
                new int[0][0], new int[0], 0, 1));
        assertThrows(IllegalArgumentException.class, () -> DiffusionGemmaGenerationSupport.computeTokensPerForward(
                filled(1, 10, 5), new int[] {1, 2}, 0, 1));
        assertThrows(IllegalArgumentException.class, () -> DiffusionGemmaGenerationSupport.computeTokensPerForward(
                filled(1, 10, 5), new int[] {0}, 0, 1));
        assertThrows(IllegalArgumentException.class, () -> DiffusionGemmaGenerationSupport.computeTokensPerForward(
                filled(1, 10, 5), new int[] {1}, 11, 1));
    }

    private static int[][] filled(int rows, int cols, int value) {
        int[][] result = new int[rows][cols];
        for (int row = 0; row < rows; row++) {
            java.util.Arrays.fill(result[row], value);
        }
        return result;
    }

    private static void fillTail(int[] row, int count, int value) {
        for (int index = row.length - count; index < row.length; index++) {
            row[index] = value;
        }
    }
}
