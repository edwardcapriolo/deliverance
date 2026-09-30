package io.teknek.deliverance.model.diffusiongemma;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class StableAndConfidentStoppingCriteriaTest {

    @Test
    void testStableAndConfidentStoppingCriteriaConfidence() {
        Lighter lighter = new Lighter(new MetricRegistry());
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2));
             TensorRef canvas = canvas(lighter, 1, 10, 7);
             TensorRef output = lighter.allocate(io.teknek.deliverance.DType.F32, TensorShape.of(1, 1))) {
            StableAndConfidentStoppingCriteria strict = criteria(lighter, 0, 1.0e-2f);
            StableAndConfidentStoppingCriteria lax = criteria(lighter, 0, 9.20f);
            StableAndConfidentStoppingCriteria tooLax = criteria(lighter, 0, 9.22f);

            try (TensorRef maxEntropy = logits(lighter, 1, 10, 10_000, 0.0f)) {
                strict.shouldStop(output, canvas, maxEntropy);
                assertEquals(0.0f, output.get(0, 0), 0.0f);
                lax.shouldStop(output, canvas, maxEntropy);
                assertEquals(0.0f, output.get(0, 0), 0.0f);
                tooLax.shouldStop(output, canvas, maxEntropy);
                assertEquals(1.0f, output.get(0, 0), 0.0f);
            }

            try (TensorRef mediumEntropy = logits(lighter, 1, 10, 10_000, 0.0f)) {
                setPreferredToken(mediumEntropy, 14.5f);
                strict.shouldStop(output, canvas, mediumEntropy);
                assertEquals(0.0f, output.get(0, 0), 0.0f);
                lax.shouldStop(output, canvas, mediumEntropy);
                assertEquals(1.0f, output.get(0, 0), 0.0f);
            }

            try (TensorRef lowEntropy = logits(lighter, 1, 10, 10_000, 0.0f)) {
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
        Lighter lighter = new Lighter(new MetricRegistry());
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2));
             TensorRef canvas1 = canvas(lighter, 1, 10, 7);
             TensorRef canvas2 = canvas(lighter, 1, 10, 11);
             TensorRef logits = logits(lighter, 1, 10, 10_000, 0.0f);
             TensorRef output = lighter.allocate(io.teknek.deliverance.DType.F32, TensorShape.of(1, 1))) {
            StableAndConfidentStoppingCriteria threshold1 = criteria(lighter, 1, 9.22f);
            StableAndConfidentStoppingCriteria threshold2 = criteria(lighter, 2, 9.22f);

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
        Lighter lighter = new Lighter(new MetricRegistry());
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2))) {
            assertThrows(IllegalArgumentException.class, () -> criteria(lighter, -1, 0.1f));
            assertThrows(IllegalArgumentException.class, () -> criteria(lighter, 0, 0.0f));
        }
    }

    private static StableAndConfidentStoppingCriteria criteria(Lighter lighter, int stabilityThreshold,
            float confidenceThreshold) {
        MetricRegistry metrics = new MetricRegistry();
        return new StableAndConfidentStoppingCriteria(stabilityThreshold, confidenceThreshold,
                lighter, metrics);
    }

    private static TensorRef canvas(Lighter lighter, int batchSize, int canvasLength, int offset) {
        TensorRef canvas = lighter.allocate(io.teknek.deliverance.DType.F32, TensorShape.of(batchSize, canvasLength));
        for (int batch = 0; batch < batchSize; batch++) {
            for (int position = 0; position < canvasLength; position++) {
                canvas.set(offset + position, batch, position);
            }
        }
        return canvas;
    }

    private static TensorRef logits(Lighter lighter, int batchSize, int canvasLength, int vocabSize, float value) {
        TensorRef logits = lighter.allocate(io.teknek.deliverance.DType.F32,
                TensorShape.of(batchSize, canvasLength, vocabSize));
        for (int batch = 0; batch < batchSize; batch++) {
            for (int position = 0; position < canvasLength; position++) {
                for (int token = 0; token < vocabSize; token++) {
                    logits.set(value, batch, position, token);
                }
            }
        }
        return logits;
    }

    private static void setPreferredToken(TensorRef logits, float value) {
        for (int batch = 0; batch < logits.shape().dim(0); batch++) {
            for (int position = 0; position < logits.shape().dim(1); position++) {
                logits.set(value, batch, position, 0);
            }
        }
    }
}
