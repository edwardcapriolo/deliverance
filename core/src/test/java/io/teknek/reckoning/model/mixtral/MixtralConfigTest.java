package io.teknek.reckoning.model.mixtral;

import io.teknek.reckoning.math.ActivationFunction;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Protects the Mixtral routing dimensions consumed by the TensorRef MoE path. */
class MixtralConfigTest {
    @Test
    void retainsExpertRoutingConfiguration() {
        MixtralConfig config = new MixtralConfig(
                128, 16, 32, 4, 2, 2, 1.0e-5f, 128, 1, 2, 10_000.0,
                ActivationFunction.Type.SILU, 8, 2);

        assertEquals(8, config.numberOfExperts);
        assertEquals(2, config.numberOfExpertsPerToken);
        assertEquals(16, config.embeddingLength);
        assertEquals(32, config.hiddenLength);
        assertEquals(2, config.numberOfKeyValueHeads);
    }
}
