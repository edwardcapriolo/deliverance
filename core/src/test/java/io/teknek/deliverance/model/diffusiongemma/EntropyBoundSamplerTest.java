package io.teknek.deliverance.model.diffusiongemma;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.dropwizard.metrics5.MetricRegistry;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class EntropyBoundSamplerTest {

    @Test
    public void testEbSamplerInitializeCanvas() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2))) {
        EntropyBoundSampler sampler = new EntropyBoundSampler(0.1f, 256, 10_000, new java.util.Random(1),
                tensorOperations(pool), new MetricRegistry());

        FloatBufferTensor canvas1 = new FloatBufferTensor(1, 256);
        FloatBufferTensor canvas2 = new FloatBufferTensor(1, 256);
        sampler.initializeCanvas(canvas1);
        sampler.initializeCanvas(canvas2);

        assertEquals(1, canvas1.shape().first());
        assertEquals(256, canvas1.shape().last());
        assertEquals(1, canvas2.shape().first());
        assertEquals(256, canvas2.shape().last());
        assertFalse(tensorEquals(canvas1, canvas2));
        for (int position = 0; position < canvas1.shape().last(); position++) {
            int token = (int) canvas1.get(0, position);
            assertTrue(token >= 0 && token < 10_000);
        }
        }
    }

    @Test
    public void testEbSamplerAcceptCanvas() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2))) {
        EntropyBoundSampler lowEntropyBound = new EntropyBoundSampler(1.0e-2f, 256, 10_000, new java.util.Random(1),
                tensorOperations(pool), new MetricRegistry());
        EntropyBoundSampler highEntropyBound = new EntropyBoundSampler(1.0e-1f, 256, 10_000, new java.util.Random(1),
                tensorOperations(pool), new MetricRegistry());
        FloatBufferTensor currentCanvas = numberedCanvas(1, 256, 100);
        FloatBufferTensor denoiserCanvas = numberedCanvas(1, 256, 1_000);
        FloatBufferTensor acceptedLow = new FloatBufferTensor(1, 256);
        FloatBufferTensor acceptedHigh = new FloatBufferTensor(1, 256);

        try (AbstractTensor logits = entropyFixtureLogits()) {
            highEntropyBound.acceptCanvas(acceptedHigh, currentCanvas, denoiserCanvas, logits, 48);
            lowEntropyBound.acceptCanvas(acceptedLow, currentCanvas, denoiserCanvas, logits, 48);

            assertEquals(countDenoiserTokens(acceptedLow, denoiserCanvas) + 1,
                    countDenoiserTokens(acceptedHigh, denoiserCanvas));
        }
        }
    }

    @Test
    public void testEbSamplerRenoiseCanvas() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2))) {
        EntropyBoundSampler sampler = new EntropyBoundSampler(1.0e-1f, 256, 10_000, new java.util.Random(1),
                tensorOperations(pool), new MetricRegistry());
        FloatBufferTensor currentCanvas = numberedCanvas(1, 256, 100);
        FloatBufferTensor denoiserCanvas = numberedCanvas(1, 256, 1_000);
        FloatBufferTensor acceptedCanvas = new FloatBufferTensor(1, 256);
        FloatBufferTensor renoisedCanvas = new FloatBufferTensor(1, 256);

        try (AbstractTensor logits = renoiseFixtureLogits()) {
            sampler.acceptCanvas(acceptedCanvas, currentCanvas, denoiserCanvas, logits, 48);
            sampler.renoiseCanvas(renoisedCanvas, acceptedCanvas, 48);

            assertTrue(countSameTokens(acceptedCanvas, renoisedCanvas) >= 10);
            for (int position = 0; position < 9; position++) {
                assertEquals(acceptedCanvas.get(0, position), renoisedCanvas.get(0, position), 0.0f);
            }
            for (int position = 0; position < 256; position++) {
                int token = (int) renoisedCanvas.get(0, position);
                assertTrue(token >= 0 && token < 10_000);
            }
        }
        }
    }

    @Test
    public void renoiseRequiresPriorAcceptCanvas() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2))) {
            EntropyBoundSampler sampler = new EntropyBoundSampler(1.0e-1f, 256, 10_000, new java.util.Random(1),
                    tensorOperations(pool), new MetricRegistry());
            assertThrows(IllegalStateException.class,
                    () -> sampler.renoiseCanvas(new FloatBufferTensor(1, 256), new FloatBufferTensor(1, 256), 48));
        }
    }

    private static PanamaTensorOperations tensorOperations(WrappedForkJoinPool pool) {
        return new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, new ArrayQueueTensorAllocator(new MetricRegistry()),
                pool);
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

    private static FloatBufferTensor numberedCanvas(int batchSize, int canvasLength, int offset) {
        FloatBufferTensor canvas = new FloatBufferTensor(batchSize, canvasLength);
        for (int batch = 0; batch < batchSize; batch++) {
            for (int position = 0; position < canvasLength; position++) {
                canvas.set(offset + position, batch, position);
            }
        }
        return canvas;
    }

    private static int countDenoiserTokens(AbstractTensor canvas, AbstractTensor denoiserCanvas) {
        int count = 0;
        for (int batch = 0; batch < canvas.shape().first(); batch++) {
            for (int position = 0; position < canvas.shape().last(); position++) {
                if (canvas.get(batch, position) == denoiserCanvas.get(batch, position)) {
                    count++;
                }
            }
        }
        return count;
    }

    private static int countSameTokens(AbstractTensor left, AbstractTensor right) {
        int count = 0;
        for (int batch = 0; batch < left.shape().first(); batch++) {
            for (int position = 0; position < left.shape().last(); position++) {
                if (left.get(batch, position) == right.get(batch, position)) {
                    count++;
                }
            }
        }
        return count;
    }

    private static boolean tensorEquals(AbstractTensor left, AbstractTensor right) {
        if (!left.shape().equals(right.shape())) {
            return false;
        }
        return countSameTokens(left, right) == left.size();
    }
}
