package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Direct QK score tests including GQA and symmetric sliding masks. */
class EmbeddingGemma2AttentionScoreOperationTest {
    @Test
    void computesGqaScoresAndMasksFutureAndPastOutsideTheSymmetricWindow() {
        Lighter lighter = new Lighter();
        EmbeddingGemma2AttentionScoreOperation operation =
                new EmbeddingGemma2AttentionScoreOperation(2, 1, 1.0f);
        try (TensorRef query0 = rowVector(lighter, 1.0f, 0.0f);
             TensorRef query1 = rowVector(lighter, 0.0f, 1.0f);
             TensorRef key = rows(lighter, new float[][] {{1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}});
             TensorRef score0 = lighter.allocate(DType.F32, TensorShape.of(3, 3));
             TensorRef score1 = lighter.allocate(DType.F32, TensorShape.of(3, 3))) {
            operation.apply(new TensorRef[] {query0, query1}, new TensorRef[] {key},
                    new TensorRef[] {score0, score1}, "sliding_attention", 1,
                    new int[] {1, 1, 1}, lighter);

            assertEquals(1.0f, score0.get(0, 0), 1.0e-6f);
            assertEquals(0.0f, score0.get(0, 1), 1.0e-6f);
            assertEquals(-Float.MAX_VALUE, score0.get(0, 2));
            assertEquals(1.0f, score1.get(1, 1), 1.0e-6f);
            assertEquals(1.0f, score1.get(1, 2), 1.0e-6f);
            assertEquals(0.0f, score1.get(1, 0), 1.0e-6f);
        }
    }

    private static TensorRef rowVector(Lighter lighter, float first, float second) {
        TensorRef result = lighter.allocate(DType.F32, TensorShape.of(3, 2));
        for (int row = 0; row < 3; row++) {
            result.set(first, row, 0);
            result.set(second, row, 1);
        }
        return result;
    }

    private static TensorRef rows(Lighter lighter, float[][] values) {
        TensorRef result = lighter.allocate(DType.F32, TensorShape.of(values.length, values[0].length));
        for (int row = 0; row < values.length; row++) {
            for (int column = 0; column < values[row].length; column++) {
                result.set(values[row][column], row, column);
            }
        }
        return result;
    }
}
