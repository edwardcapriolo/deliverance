package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** First-principles Q/K/V projection and head-normalization tests. */
class EmbeddingGemma2AttentionProjectionOperationTest {
    @Test
    void projectsAndNormalizesQueryKeyAndValueHeads() {
        int hidden = 3;
        int queryHeads = 2;
        int keyValueHeads = 1;
        int headWidth = 2;
        float epsilon = 1.0e-6f;
        Lighter lighter = new Lighter();
        EmbeddingGemma2AttentionProjectionOperation operation =
                new EmbeddingGemma2AttentionProjectionOperation(hidden, queryHeads, keyValueHeads, headWidth, epsilon);
        try (TensorRef input = lighter.allocate(DType.F32, TensorShape.of(1, hidden));
             TensorRef queryProjection = lighter.allocate(DType.F32, TensorShape.of(4, hidden));
             TensorRef keyProjection = lighter.allocate(DType.F32, TensorShape.of(2, hidden));
             TensorRef valueProjection = lighter.allocate(DType.F32, TensorShape.of(2, hidden));
             TensorRef query = lighter.allocate(DType.F32, TensorShape.of(1, 4));
             TensorRef key = lighter.allocate(DType.F32, TensorShape.of(1, 2));
             TensorRef value = lighter.allocate(DType.F32, TensorShape.of(1, 2));
             TensorRef queryNorm = lighter.allocate(DType.F32, TensorShape.of(1, headWidth));
             TensorRef keyNorm = lighter.allocate(DType.F32, TensorShape.of(1, headWidth))) {
            input.set(1.0f, 0, 0);
            input.set(2.0f, 0, 1);
            input.set(3.0f, 0, 2);
            fillProjection(queryProjection, 1.0f);
            fillProjection(keyProjection, 0.5f);
            fillProjection(valueProjection, -1.0f);
            queryNorm.set(1.0f, 0, 0);
            queryNorm.set(0.5f, 0, 1);
            keyNorm.set(2.0f, 0, 0);
            keyNorm.set(-1.0f, 0, 1);

            operation.apply(query, key, value, input, queryProjection, keyProjection, valueProjection,
                    queryNorm, keyNorm, lighter);

            assertNormalizedProjected(query, queryProjection, input, queryNorm, 2, 2, epsilon);
            assertNormalizedProjected(key, keyProjection, input, keyNorm, 1, 2, epsilon);
            assertNormalizedProjected(value, valueProjection, input, null, 1, 2, epsilon);
        }
    }

    private static void fillProjection(TensorRef projection, float value) {
        for (int row = 0; row < projection.shape().first(); row++) {
            for (int column = 0; column < projection.shape().last(); column++) {
                projection.set(value * (row + column + 1), row, column);
            }
        }
    }

    private static void assertNormalizedProjected(TensorRef output, TensorRef projection, TensorRef input,
                                                  TensorRef norm, int heads, int headWidth, float epsilon) {
        for (int head = 0; head < heads; head++) {
            int offset = head * headWidth;
            float[] projected = new float[headWidth];
            double sumSquares = 0.0;
            for (int column = 0; column < headWidth; column++) {
                float value = 0.0f;
                for (int inputColumn = 0; inputColumn < input.shape().last(); inputColumn++) {
                    value += input.get(0, inputColumn) * projection.get(offset + column, inputColumn);
                }
                projected[column] = value;
                sumSquares += value * value;
            }
            float inverseRms = (float) (1.0 / Math.sqrt(sumSquares / headWidth + epsilon));
            for (int column = 0; column < headWidth; column++) {
                float expected = projected[column] * inverseRms;
                if (norm != null) {
                    expected *= norm.get(0, column);
                }
                assertEquals(expected, output.get(0, offset + column), 1.0e-5f);
            }
        }
    }
}
