package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/** First-principles tests for EmbeddingGemma 2 scaled token embeddings. */
class EmbeddingGemma2ScaledWordEmbeddingOperationTest {
    @Test
    void gathersRowsAndAppliesEmbeddingScale() {
        EmbeddingGemma2ScaledWordEmbeddingOperation operation =
                new EmbeddingGemma2ScaledWordEmbeddingOperation();
        Lighter lighter = new Lighter();
        try (TensorRef table = lighter.allocate(DType.F32, TensorShape.of(3, 4));
             TensorRef output = lighter.allocate(DType.F32, TensorShape.of(2, 4))) {
            fill(table, new float[][] {
                    {1.0f, 2.0f, 3.0f, 4.0f},
                    {-1.0f, 0.5f, 2.0f, -3.0f},
                    {4.0f, 3.0f, 2.0f, 1.0f}
            });

            operation.apply(output, table, new int[] {2, 0}, 2.0f, lighter);

            assertEquals(8.0f, output.get(0, 0));
            assertEquals(6.0f, output.get(0, 1));
            assertEquals(4.0f, output.get(0, 2));
            assertEquals(2.0f, output.get(0, 3));
            assertEquals(2.0f, output.get(1, 0));
            assertEquals(4.0f, output.get(1, 1));
            assertEquals(6.0f, output.get(1, 2));
            assertEquals(8.0f, output.get(1, 3));
        }
    }

    @Test
    void rejectsTokenOutsideVocabulary() {
        EmbeddingGemma2ScaledWordEmbeddingOperation operation =
                new EmbeddingGemma2ScaledWordEmbeddingOperation();
        Lighter lighter = new Lighter();
        try (TensorRef table = lighter.allocate(DType.F32, TensorShape.of(2, 3));
             TensorRef output = lighter.allocate(DType.F32, TensorShape.of(1, 3))) {
            assertThrows(IllegalArgumentException.class,
                    () -> operation.apply(output, table, new int[] {2}, 1.0f, lighter));
        }
    }

    private static void fill(TensorRef tensor, float[][] values) {
        for (int row = 0; row < values.length; row++) {
            for (int column = 0; column < values[row].length; column++) {
                tensor.set(values[row][column], row, column);
            }
        }
    }
}
