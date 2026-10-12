package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Direct test for the bias-free GELU gated MLP ordering. */
class EmbeddingGemma2MlpOperationTest {
    @Test
    void appliesGateActivationUpAndDownInReferenceOrder() {
        Lighter lighter = new Lighter();
        try (TensorRef input = matrix(lighter, new float[][] {{1.0f, 2.0f}});
             TensorRef gate = matrix(lighter, new float[][] {{1.0f, 0.0f}});
             TensorRef up = matrix(lighter, new float[][] {{2.0f, 0.0f}});
             TensorRef down = matrix(lighter, new float[][] {{3.0f}, {0.0f}});
             TensorRef output = lighter.allocate(DType.F32, TensorShape.of(1, 2))) {
            new EmbeddingGemma2MlpOperation(2, 1).apply(output, input, gate, up, down, lighter);
            float activated = (float) (0.5 * 1.0 * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI)
                    * (1.0 + 0.044715 * 1.0))));
            assertEquals(3.0f * activated * 2.0f, output.get(0, 0), 1.0e-5f);
            assertEquals(0.0f, output.get(0, 1), 1.0e-6f);
        }
    }

    private static TensorRef matrix(Lighter lighter, float[][] values) {
        TensorRef result = lighter.allocate(DType.F32, TensorShape.of(values.length, values[0].length));
        for (int row = 0; row < values.length; row++) {
            for (int column = 0; column < values[row].length; column++) {
                result.set(values[row][column], row, column);
            }
        }
        return result;
    }
}
