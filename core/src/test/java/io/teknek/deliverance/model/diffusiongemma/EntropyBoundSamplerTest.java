package io.teknek.deliverance.model.diffusiongemma;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class EntropyBoundSamplerTest {

    @Test
    public void initializeCanvasCreatesRandomTokenIdsInRange() {
        EntropyBoundSampler sampler = new EntropyBoundSampler(0.1f, 256, 10_000, new java.util.Random(1));

        int[][] canvas1 = sampler.initializeCanvas(1);
        int[][] canvas2 = sampler.initializeCanvas(1);

        assertEquals(1, canvas1.length);
        assertEquals(256, canvas1[0].length);
        assertEquals(1, canvas2.length);
        assertEquals(256, canvas2[0].length);
        assertFalse(java.util.Arrays.equals(canvas1[0], canvas2[0]));
        for (int token : canvas1[0]) {
            assertTrue(token >= 0 && token < 10_000);
        }
    }

    @Test
    public void acceptCanvasAcceptsMoreLowEntropyTokensWithHigherEntropyBound() {
        EntropyBoundSampler lowEntropyBound = new EntropyBoundSampler(1.0e-2f, 256, 10_000, new java.util.Random(1));
        EntropyBoundSampler highEntropyBound = new EntropyBoundSampler(1.0e-1f, 256, 10_000, new java.util.Random(1));
        int[][] currentCanvas = numberedCanvas(1, 256, 100);
        int[][] denoiserCanvas = numberedCanvas(1, 256, 1_000);

        try (AbstractTensor logits = entropyFixtureLogits()) {
            int[][] acceptedHigh = highEntropyBound.acceptCanvas(currentCanvas, denoiserCanvas, logits, 48);
            int[][] acceptedLow = lowEntropyBound.acceptCanvas(currentCanvas, denoiserCanvas, logits, 48);

            assertEquals(countDenoiserTokens(acceptedLow, denoiserCanvas) + 1,
                    countDenoiserTokens(acceptedHigh, denoiserCanvas));
        }
    }

    @Test
    public void renoiseCanvasIsNotImplementedYet() {
        EntropyBoundSampler sampler = new EntropyBoundSampler(1.0e-1f, 256, 10_000, new java.util.Random(1));
        int[][] currentCanvas = numberedCanvas(1, 256, 100);
        int[][] denoiserCanvas = numberedCanvas(1, 256, 1_000);

        try (AbstractTensor logits = renoiseFixtureLogits()) {
            int[][] acceptedCanvas = sampler.acceptCanvas(currentCanvas, denoiserCanvas, logits, 48);
            assertThrows(UnsupportedOperationException.class, () -> sampler.renoiseCanvas(acceptedCanvas, 48));
        }
    }

    private static AbstractTensor entropyFixtureLogits() {
        FloatBufferTensor logits = new FloatBufferTensor(1, 256, 10_000);
        logits.set(18.0f, 0, 0, 0);
        logits.set(14.5f, 0, 1, 1);
        logits.set(14.5f, 0, 2, 2);
        return logits;
    }

    private static AbstractTensor renoiseFixtureLogits() {
        FloatBufferTensor logits = new FloatBufferTensor(1, 256, 10_000);
        for (int position = 0; position < 9; position++) {
            logits.set(1.0e6f, 0, position, 0);
        }
        return logits;
    }

    private static int[][] numberedCanvas(int batchSize, int canvasLength, int offset) {
        int[][] canvas = new int[batchSize][canvasLength];
        for (int batch = 0; batch < batchSize; batch++) {
            for (int position = 0; position < canvasLength; position++) {
                canvas[batch][position] = offset + position;
            }
        }
        return canvas;
    }

    private static int countDenoiserTokens(int[][] canvas, int[][] denoiserCanvas) {
        int count = 0;
        for (int batch = 0; batch < canvas.length; batch++) {
            for (int position = 0; position < canvas[batch].length; position++) {
                if (canvas[batch][position] == denoiserCanvas[batch][position]) {
                    count++;
                }
            }
        }
        return count;
    }
}
