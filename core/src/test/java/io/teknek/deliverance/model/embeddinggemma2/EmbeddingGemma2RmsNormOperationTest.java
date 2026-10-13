package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** First-principles tests for the EmbeddingGemma 2 RMSNorm contract. */
class EmbeddingGemma2RmsNormOperationTest {
    @Test
    void appliesFloat32RmsNormAndLearnedScale() {
        float epsilon = 1.0e-6f;
        EmbeddingGemma2RmsNormOperation norm = new EmbeddingGemma2RmsNormOperation(epsilon);
        Lighter lighter = new Lighter();
        try (TensorRef input = lighter.allocate(DType.F32, TensorShape.of(2, 4));
             TensorRef weight = lighter.allocate(DType.F32, TensorShape.of(1, 4));
             TensorRef output = lighter.allocate(DType.F32, TensorShape.of(2, 4))) {
            fill(input, new float[][] {{1.0f, 2.0f, 3.0f, 4.0f}, {-2.0f, 0.5f, 1.5f, 3.0f}});
            fill(weight, new float[][] {{1.0f, 0.5f, 2.0f, -1.0f}});

            norm.apply(output, input, weight, lighter);

            assertReference(input, weight, output, epsilon);
        }
    }

    @Test
    void supportsReferenceWithScaleFalseMode() {
        float epsilon = 1.0e-6f;
        EmbeddingGemma2RmsNormOperation norm = new EmbeddingGemma2RmsNormOperation(epsilon);
        Lighter lighter = new Lighter();
        try (TensorRef input = lighter.allocate(DType.F32, TensorShape.of(1, 3));
             TensorRef output = lighter.allocate(DType.F32, TensorShape.of(1, 3))) {
            input.set(2.0f, 0, 0);
            input.set(4.0f, 0, 1);
            input.set(4.0f, 0, 2);

            norm.apply(output, input, null, lighter);

            float inverseRms = (float) (1.0 / Math.sqrt((4.0 + 16.0 + 16.0) / 3.0 + epsilon));
            assertEquals(2.0f * inverseRms, output.get(0, 0), 1.0e-6f);
            assertEquals(4.0f * inverseRms, output.get(0, 1), 1.0e-6f);
            assertEquals(4.0f * inverseRms, output.get(0, 2), 1.0e-6f);
        }
    }

    private static void fill(TensorRef tensor, float[][] values) {
        for (int row = 0; row < values.length; row++) {
            for (int column = 0; column < values[row].length; column++) {
                tensor.set(values[row][column], row, column);
            }
        }
    }

    private static void assertReference(TensorRef input, TensorRef weight, TensorRef output, float epsilon) {
        for (int row = 0; row < input.shape().first(); row++) {
            double sumSquares = 0.0;
            for (int column = 0; column < input.shape().last(); column++) {
                float value = input.get(row, column);
                sumSquares += value * value;
            }
            float inverseRms = (float) (1.0 / Math.sqrt(sumSquares / input.shape().last() + epsilon));
            for (int column = 0; column < input.shape().last(); column++) {
                assertEquals(input.get(row, column) * inverseRms * weight.get(0, column),
                        output.get(row, column), 1.0e-6f, "row=" + row + " column=" + column);
            }
        }
    }
}
