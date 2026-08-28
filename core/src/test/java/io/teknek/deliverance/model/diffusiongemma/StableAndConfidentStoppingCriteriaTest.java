package io.teknek.deliverance.model.diffusiongemma;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class StableAndConfidentStoppingCriteriaTest {

    @Test
    void testStableAndConfidentStoppingCriteriaConfidence() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2));
             FloatBufferTensor canvas = canvas(1, 10, 7);
             FloatBufferTensor output = new FloatBufferTensor(1, 1)) {
            StableAndConfidentStoppingCriteria strict = criteria(pool, 0, 1.0e-2f);
            StableAndConfidentStoppingCriteria lax = criteria(pool, 0, 9.20f);
            StableAndConfidentStoppingCriteria tooLax = criteria(pool, 0, 9.22f);

            try (FloatBufferTensor maxEntropy = logits(1, 10, 10_000, 0.0f)) {
                strict.shouldStop(output, canvas, maxEntropy);
                assertEquals(0.0f, output.get(0, 0), 0.0f);
                lax.shouldStop(output, canvas, maxEntropy);
                assertEquals(0.0f, output.get(0, 0), 0.0f);
                tooLax.shouldStop(output, canvas, maxEntropy);
                assertEquals(1.0f, output.get(0, 0), 0.0f);
            }

            try (FloatBufferTensor mediumEntropy = logits(1, 10, 10_000, 0.0f)) {
                setPreferredToken(mediumEntropy, 14.5f);
                strict.shouldStop(output, canvas, mediumEntropy);
                assertEquals(0.0f, output.get(0, 0), 0.0f);
                lax.shouldStop(output, canvas, mediumEntropy);
                assertEquals(1.0f, output.get(0, 0), 0.0f);
            }

            try (FloatBufferTensor lowEntropy = logits(1, 10, 10_000, 0.0f)) {
                setPreferredToken(lowEntropy, 18.0f);
                strict.shouldStop(output, canvas, lowEntropy);
                assertEquals(1.0f, output.get(0, 0), 0.0f);
                lax.shouldStop(output, canvas, lowEntropy);
                assertEquals(1.0f, output.get(0, 0), 0.0f);
            }
        }
    }

    @Test
    void testStableAndConfidentStoppingCriteriaStability() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2));
             FloatBufferTensor canvas1 = canvas(1, 10, 7);
             FloatBufferTensor canvas2 = canvas(1, 10, 11);
             FloatBufferTensor logits = logits(1, 10, 10_000, 0.0f);
             FloatBufferTensor output = new FloatBufferTensor(1, 1)) {
            StableAndConfidentStoppingCriteria threshold1 = criteria(pool, 1, 9.22f);
            StableAndConfidentStoppingCriteria threshold2 = criteria(pool, 2, 9.22f);

            threshold1.shouldStop(output, canvas1, logits);
            assertEquals(0.0f, output.get(0, 0), 0.0f);
            threshold2.shouldStop(output, canvas1, logits);
            assertEquals(0.0f, output.get(0, 0), 0.0f);

            threshold1.shouldStop(output, canvas1, logits);
            assertEquals(1.0f, output.get(0, 0), 0.0f);
            threshold2.shouldStop(output, canvas1, logits);
            assertEquals(0.0f, output.get(0, 0), 0.0f);

            threshold1.shouldStop(output, canvas1, logits);
            assertEquals(1.0f, output.get(0, 0), 0.0f);
            threshold2.shouldStop(output, canvas1, logits);
            assertEquals(1.0f, output.get(0, 0), 0.0f);

            threshold1.shouldStop(output, canvas2, logits);
            assertEquals(0.0f, output.get(0, 0), 0.0f);
            threshold2.shouldStop(output, canvas2, logits);
            assertEquals(0.0f, output.get(0, 0), 0.0f);
        }
    }

    @Test
    void validatesParameters() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2))) {
            assertThrows(IllegalArgumentException.class, () -> criteria(pool, -1, 0.1f));
            assertThrows(IllegalArgumentException.class, () -> criteria(pool, 0, 0.0f));
        }
    }

    private static StableAndConfidentStoppingCriteria criteria(WrappedForkJoinPool pool, int stabilityThreshold,
            float confidenceThreshold) {
        MetricRegistry metrics = new MetricRegistry();
        return new StableAndConfidentStoppingCriteria(stabilityThreshold, confidenceThreshold,
                new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, new ArrayQueueTensorAllocator(metrics), pool),
                metrics);
    }

    private static FloatBufferTensor canvas(int batchSize, int canvasLength, int offset) {
        FloatBufferTensor canvas = new FloatBufferTensor(batchSize, canvasLength);
        for (int batch = 0; batch < batchSize; batch++) {
            for (int position = 0; position < canvasLength; position++) {
                canvas.set(offset + position, batch, position);
            }
        }
        return canvas;
    }

    private static FloatBufferTensor logits(int batchSize, int canvasLength, int vocabSize, float value) {
        FloatBufferTensor logits = new FloatBufferTensor(batchSize, canvasLength, vocabSize);
        for (int batch = 0; batch < batchSize; batch++) {
            for (int position = 0; position < canvasLength; position++) {
                for (int token = 0; token < vocabSize; token++) {
                    logits.set(value, batch, position, token);
                }
            }
        }
        return logits;
    }

    private static void setPreferredToken(FloatBufferTensor logits, float value) {
        for (int batch = 0; batch < logits.shape().dim(0); batch++) {
            for (int position = 0; position < logits.shape().dim(1); position++) {
                logits.set(value, batch, position, 0);
            }
        }
    }
}
