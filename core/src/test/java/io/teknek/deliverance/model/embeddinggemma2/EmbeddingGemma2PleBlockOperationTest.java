package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** First-principles end-to-end tests for one PLE residual block. */
class EmbeddingGemma2PleBlockOperationTest {
    @Test
    void matchesReferenceForIdentityProjections() {
        int hidden = 2;
        float epsilon = 1.0e-6f;
        Lighter lighter = new Lighter();
        EmbeddingGemma2PleBlockOperation operation = new EmbeddingGemma2PleBlockOperation(hidden, epsilon);
        try (TensorRef hiddenStates = lighter.allocate(DType.F32, TensorShape.of(2, hidden));
             TensorRef perLayerInput = lighter.allocate(DType.F32, TensorShape.of(2, hidden));
             TensorRef gateWeight = lighter.allocate(DType.F32, TensorShape.of(hidden, hidden));
             TensorRef projectionWeight = lighter.allocate(DType.F32, TensorShape.of(hidden, hidden));
             TensorRef normWeight = lighter.allocate(DType.F32, TensorShape.of(1, hidden));
             TensorRef output = lighter.allocate(DType.F32, TensorShape.of(2, hidden))) {
            fill(hiddenStates, new float[][] {{1.0f, -2.0f}, {0.5f, 3.0f}});
            fill(perLayerInput, new float[][] {{2.0f, 0.5f}, {-1.0f, 2.0f}});
            identity(gateWeight);
            identity(projectionWeight);
            normWeight.set(1.0f, 0, 0);
            normWeight.set(0.5f, 0, 1);

            operation.apply(output, hiddenStates, perLayerInput, gateWeight, projectionWeight, normWeight, lighter);

            for (int row = 0; row < 2; row++) {
                float[] gated = new float[hidden];
                for (int column = 0; column < hidden; column++) {
                    gated[column] = ActivationFunction.eval(ActivationFunction.Type.GELU_PYTORCH_TANH,
                            hiddenStates.get(row, column)) * perLayerInput.get(row, column);
                }
                double sumSquares = 0.0;
                for (float value : gated) {
                    sumSquares += value * value;
                }
                float inverseRms = (float) (1.0 / Math.sqrt(sumSquares / hidden + epsilon));
                for (int column = 0; column < hidden; column++) {
                    float normalized = gated[column] * inverseRms * normWeight.get(0, column);
                    assertEquals(normalized + hiddenStates.get(row, column), output.get(row, column), 1.0e-5f);
                }
            }
        }
    }

    private static void identity(TensorRef tensor) {
        tensor.set(1.0f, 0, 0);
        tensor.set(1.0f, 1, 1);
    }

    private static void fill(TensorRef tensor, float[][] values) {
        for (int row = 0; row < values.length; row++) {
            for (int column = 0; column < values[row].length; column++) {
                tensor.set(values[row][column], row, column);
            }
        }
    }
}
