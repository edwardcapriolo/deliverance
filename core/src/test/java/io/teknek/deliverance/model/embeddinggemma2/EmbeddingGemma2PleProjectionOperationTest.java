package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** First-principles tests for the EmbeddingGemma 2 PLE projection. */
class EmbeddingGemma2PleProjectionOperationTest {
    @Test
    void projectsRowsScalesByInverseSqrtInputWidthAndPreservesRowOrder() {
        EmbeddingGemma2PleProjectionOperation operation =
                new EmbeddingGemma2PleProjectionOperation(2, 2, 2);
        Lighter lighter = new Lighter();
        try (TensorRef input = lighter.allocate(DType.F32, TensorShape.of(2, 2));
             TensorRef projection = lighter.allocate(DType.F32, TensorShape.of(4, 2));
             TensorRef output = lighter.allocate(DType.F32, TensorShape.of(2, 4))) {
            input.set(1.0f, 0, 0);
            input.set(2.0f, 0, 1);
            input.set(3.0f, 1, 0);
            input.set(4.0f, 1, 1);
            for (int row = 0; row < 4; row++) {
                projection.set(row + 1.0f, row, 0);
                projection.set((row + 1.0f) * 2.0f, row, 1);
            }

            operation.apply(output, input, projection, lighter);

            float scale = (float) (1.0 / Math.sqrt(2.0));
            for (int row = 0; row < 2; row++) {
                float first = input.get(row, 0);
                float second = input.get(row, 1);
                for (int column = 0; column < 4; column++) {
                    float expected = (first * projection.get(column, 0)
                            + second * projection.get(column, 1)) * scale;
                    assertEquals(expected, output.get(row, column), 1.0e-5f,
                            "row=" + row + " column=" + column);
                }
            }
        }
    }
}
