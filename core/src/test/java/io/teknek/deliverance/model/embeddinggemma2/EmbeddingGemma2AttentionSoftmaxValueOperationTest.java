package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Direct tests for masked attention normalization and GQA value aggregation. */
class EmbeddingGemma2AttentionSoftmaxValueOperationTest {
    @Test
    void normalizesRowsAndKeepsMaskedEntriesAtZero() {
        Lighter lighter = new Lighter();
        try (TensorRef scores = tensor(lighter, new float[][] {
                {1.0f, 2.0f, -Float.MAX_VALUE},
                {-Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE}
        })) {
            new EmbeddingGemma2AttentionSoftmaxOperation().apply(new TensorRef[] {scores}, lighter);
            assertEquals(1.0f / (1.0f + (float) Math.exp(1.0)), scores.get(0, 0), 1.0e-6f);
            assertEquals((float) Math.exp(1.0) / (1.0f + (float) Math.exp(1.0)), scores.get(0, 1), 1.0e-6f);
            assertEquals(0.0f, scores.get(0, 2), 1.0e-6f);
            assertEquals(0.0f, scores.get(1, 0), 1.0e-6f);
        }
    }

    @Test
    void aggregatesValuesUsingGroupedQueryHeads() {
        Lighter lighter = new Lighter();
        try (TensorRef scores0 = tensor(lighter, new float[][] {{1.0f, 0.0f}, {0.0f, 1.0f}});
             TensorRef scores1 = tensor(lighter, new float[][] {{0.0f, 1.0f}, {1.0f, 0.0f}});
             TensorRef values = tensor(lighter, new float[][] {{10.0f, 1.0f}, {20.0f, 2.0f}});
             TensorRef output = lighter.allocate(DType.F32, TensorShape.of(2, 4))) {
            new EmbeddingGemma2AttentionValueOperation(2, 1, 2)
                    .apply(new TensorRef[] {scores0, scores1}, values, output, lighter);
            assertEquals(10.0f, output.get(0, 0), 1.0e-6f);
            assertEquals(1.0f, output.get(0, 1), 1.0e-6f);
            assertEquals(20.0f, output.get(0, 2), 1.0e-6f);
            assertEquals(2.0f, output.get(0, 3), 1.0e-6f);
            assertEquals(20.0f, output.get(1, 0), 1.0e-6f);
            assertEquals(10.0f, output.get(1, 2), 1.0e-6f);
        }
    }

    private static TensorRef tensor(Lighter lighter, float[][] values) {
        TensorRef result = lighter.allocate(DType.F32, TensorShape.of(values.length, values[0].length));
        for (int row = 0; row < values.length; row++) {
            for (int column = 0; column < values[row].length; column++) {
                result.set(values[row][column], row, column);
            }
        }
        return result;
    }
}
