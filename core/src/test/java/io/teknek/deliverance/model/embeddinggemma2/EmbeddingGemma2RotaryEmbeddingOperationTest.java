package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** First-principles RoPE tests, including layer-type-specific theta values. */
class EmbeddingGemma2RotaryEmbeddingOperationTest {
    @Test
    void appliesPositionRotationWithConfiguredTheta() {
        Lighter lighter = new Lighter();
        EmbeddingGemma2RotaryEmbeddingOperation operation =
                new EmbeddingGemma2RotaryEmbeddingOperation(4, 1, 10_000.0, 4);
        try (TensorRef target = lighter.allocate(DType.F32, TensorShape.of(2, 4))) {
            target.set(1.0f, 0, 0);
            target.set(0.0f, 0, 1);
            target.set(1.0f, 0, 2);
            target.set(0.0f, 0, 3);
            target.set(0.0f, 1, 0);
            target.set(1.0f, 1, 1);
            target.set(0.0f, 1, 2);
            target.set(0.0f, 1, 3);

            operation.apply(target, 0, lighter);

            assertEquals(1.0f, target.get(0, 0), 1.0e-6f);
            assertEquals(0.0f, target.get(0, 1), 1.0e-6f);
            assertEquals(1.0f, target.get(0, 2), 1.0e-6f);
            assertEquals(0.0f, target.get(0, 3), 1.0e-6f);
            double angle = 1.0 / Math.pow(10_000.0, 2.0 / 4.0);
            assertEquals(0.0f, target.get(1, 0), 1.0e-6f);
            assertEquals(Math.cos(angle), target.get(1, 1), 1.0e-6f);
            assertEquals(0.0f, target.get(1, 2), 1.0e-6f);
            assertEquals(Math.sin(angle), target.get(1, 3), 1.0e-6f);
        }
    }
}
