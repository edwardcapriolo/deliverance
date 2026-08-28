package io.teknek.deliverance.tensor;

import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TensorNormalizationTest {

    @Test
    void rmsNormAppliesRowWiseScale() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2));
             FloatBufferTensor input = new FloatBufferTensor(2, 4);
             FloatBufferTensor weight = new FloatBufferTensor(1, 4);
             FloatBufferTensor output = new FloatBufferTensor(2, 4)) {
            fill(input, new float[][] {{1.0f, 2.0f, 3.0f, 4.0f}, {-2.0f, 0.5f, 1.5f, 3.0f}});
            fill(weight, new float[][] {{1.0f, 0.5f, 2.0f, -1.0f}});

            TensorNormalization.rmsNorm(output, input, weight, 1.0e-6f, new NaiveTensorOperations(), pool);

            assertRmsNorm(input, weight, output, 1.0e-6f);
        }
    }

    @Test
    void rmsNormSupportsNoScaleWeight() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2));
             FloatBufferTensor input = new FloatBufferTensor(1, 3);
             FloatBufferTensor output = new FloatBufferTensor(1, 3)) {
            fill(input, new float[][] {{2.0f, 4.0f, 4.0f}});

            TensorNormalization.rmsNorm(output, input, null, 1.0e-6f, new NaiveTensorOperations(), pool);

            assertRmsNorm(input, null, output, 1.0e-6f);
        }
    }

    @Test
    void rmsNormLastDimAppliesRowWiseScaleAcrossBatchAndCanvas() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2));
             FloatBufferTensor input = new FloatBufferTensor(2, 3, 4);
             FloatBufferTensor weight = new FloatBufferTensor(1, 4);
             FloatBufferTensor output = new FloatBufferTensor(2, 3, 4)) {
            int value = 1;
            for (int batch = 0; batch < 2; batch++) {
                for (int position = 0; position < 3; position++) {
                    for (int hidden = 0; hidden < 4; hidden++) {
                        input.set(value++, batch, position, hidden);
                    }
                }
            }
            fill(weight, new float[][] {{1.0f, 0.5f, 2.0f, -1.0f}});

            TensorNormalization.rmsNormLastDim(output, input, weight, 1.0e-6f, new NaiveTensorOperations(), pool);

            assertRmsNormLastDim(input, weight, output, 1.0e-6f);
        }
    }

    @Test
    void rmsNormLastDimSupportsNoScaleWeight() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2));
             FloatBufferTensor input = new FloatBufferTensor(1, 2, 3);
             FloatBufferTensor output = new FloatBufferTensor(1, 2, 3)) {
            input.set(1.0f, 0, 0, 0);
            input.set(2.0f, 0, 0, 1);
            input.set(3.0f, 0, 0, 2);
            input.set(4.0f, 0, 1, 0);
            input.set(5.0f, 0, 1, 1);
            input.set(6.0f, 0, 1, 2);

            TensorNormalization.rmsNormLastDim(output, input, null, 1.0e-6f, new NaiveTensorOperations(), pool);

            assertRmsNormLastDim(input, null, output, 1.0e-6f);
        }
    }

    @Test
    void rmsNormRejectsNon2dInput() {
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(new java.util.concurrent.ForkJoinPool(2));
             FloatBufferTensor input = new FloatBufferTensor(1, 2, 3);
             FloatBufferTensor output = new FloatBufferTensor(1, 2, 3)) {
            assertThrows(IllegalArgumentException.class,
                    () -> TensorNormalization.rmsNorm(output, input, null, 1.0e-6f, new NaiveTensorOperations(), pool));
        }
    }

    private static void fill(FloatBufferTensor tensor, float[][] values) {
        for (int row = 0; row < values.length; row++) {
            for (int col = 0; col < values[row].length; col++) {
                tensor.set(values[row][col], row, col);
            }
        }
    }

    private static void assertRmsNorm(AbstractTensor input, AbstractTensor weight, AbstractTensor output, float eps) {
        int hidden = (int) input.shape().last();
        for (int row = 0; row < input.shape().first(); row++) {
            double sumSquares = 0.0;
            for (int col = 0; col < hidden; col++) {
                float value = input.get(row, col);
                sumSquares += value * value;
            }
            float invRms = (float) (1.0 / Math.sqrt(sumSquares / hidden + eps));
            for (int col = 0; col < hidden; col++) {
                float expected = input.get(row, col) * invRms;
                if (weight != null) {
                    expected *= weight.get(0, col);
                }
                assertEquals(expected, output.get(row, col), 1.0e-6f, "row=" + row + " col=" + col);
            }
        }
    }

    private static void assertRmsNormLastDim(AbstractTensor input, AbstractTensor weight, AbstractTensor output,
            float eps) {
        int hidden = (int) input.shape().dim(2);
        for (int batch = 0; batch < input.shape().dim(0); batch++) {
            for (int position = 0; position < input.shape().dim(1); position++) {
                double sumSquares = 0.0;
                for (int col = 0; col < hidden; col++) {
                    float value = input.get(batch, position, col);
                    sumSquares += value * value;
                }
                float invRms = (float) (1.0 / Math.sqrt(sumSquares / hidden + eps));
                for (int col = 0; col < hidden; col++) {
                    float expected = input.get(batch, position, col) * invRms;
                    if (weight != null) {
                        expected *= weight.get(0, col);
                    }
                    assertEquals(expected, output.get(batch, position, col), 1.0e-6f,
                            "batch=" + batch + " position=" + position + " hidden=" + col);
                }
            }
        }
    }
}
