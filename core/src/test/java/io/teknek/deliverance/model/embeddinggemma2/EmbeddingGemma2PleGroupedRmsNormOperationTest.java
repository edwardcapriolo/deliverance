package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Proves that PLE slices normalize independently without a reshape or copy. */
class EmbeddingGemma2PleGroupedRmsNormOperationTest {
    @Test
    void normalizesEachLayerSliceIndependently() {
        float epsilon = 1.0e-6f;
        Lighter lighter = new Lighter();
        EmbeddingGemma2PleGroupedRmsNormOperation operation =
                new EmbeddingGemma2PleGroupedRmsNormOperation(2, 3, epsilon);
        try (TensorRef target = lighter.allocate(DType.F32, TensorShape.of(2, 6));
             TensorRef weights = lighter.allocate(DType.F32, TensorShape.of(1, 3))) {
            fill(target, new float[][] {{1.0f, 2.0f, 2.0f, 3.0f, 4.0f, 12.0f},
                    {-2.0f, 1.0f, 2.0f, 6.0f, 8.0f, 10.0f}});
            fill(weights, new float[][] {{1.0f, 0.5f, -1.0f}});

            operation.apply(target, weights, lighter);

            assertReference(target, new float[][] {{1.0f, 2.0f, 2.0f, 3.0f, 4.0f, 12.0f},
                    {-2.0f, 1.0f, 2.0f, 6.0f, 8.0f, 10.0f}}, weights, epsilon);
        }
    }

    private static void fill(TensorRef tensor, float[][] values) {
        for (int row = 0; row < values.length; row++) {
            for (int column = 0; column < values[row].length; column++) {
                tensor.set(values[row][column], row, column);
            }
        }
    }

    private static void assertReference(TensorRef actual, float[][] input, TensorRef weights, float epsilon) {
        for (int row = 0; row < input.length; row++) {
            for (int group = 0; group < 2; group++) {
                int offset = group * 3;
                double sumSquares = 0.0;
                for (int column = 0; column < 3; column++) {
                    sumSquares += input[row][offset + column] * input[row][offset + column];
                }
                float inverseRms = (float) (1.0 / Math.sqrt(sumSquares / 3.0 + epsilon));
                for (int column = 0; column < 3; column++) {
                    assertEquals(input[row][offset + column] * inverseRms * weights.get(0, column),
                            actual.get(row, offset + column), 1.0e-5f,
                            "row=" + row + " group=" + group + " column=" + column);
                }
            }
        }
    }
}
