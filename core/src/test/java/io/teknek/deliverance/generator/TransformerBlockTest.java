package io.teknek.deliverance.generator;

import io.teknek.deliverance.tensor.TensorTestSupport;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TransformerBlockTest {

    @Test
    void applyResidualRangeAddsResidualWithOptionalMultiplier() {
        try (var target = TensorTestSupport.tensorOf(2, 3, 1, 2, 3, 4, 5, 6);
             var residual = TensorTestSupport.tensorOf(2, 3, 10, 20, 30, 40, 50, 60)) {

            TransformerBlock.applyResidualRange(target, residual, 0.5f, 1, 4);

            assertEquals(1.0f, target.get(0, 0), 1.0e-6f);
            assertEquals(21.0f, target.get(0, 1), 1.0e-6f);
            assertEquals(31.5f, target.get(0, 2), 1.0e-6f);
            assertEquals(42.0f, target.get(1, 0), 1.0e-6f);
            assertEquals(52.5f, target.get(1, 1), 1.0e-6f);
            assertEquals(6.0f, target.get(1, 2), 1.0e-6f);
        }
    }
}
