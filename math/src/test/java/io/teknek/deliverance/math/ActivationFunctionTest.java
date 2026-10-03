package io.teknek.deliverance.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Validate the behavior of the {@code ActivationFunction} class. Specifically, it tests the correctness
 * of the various activation functions implemented in the {@code ActivationFunction.Type} enum and their
 * mapped evaluations.
 */
class ActivationFunctionTest {

    @Test
    void everyActivationMapsZeroToZero() {
        for (ActivationFunction.Type type : ActivationFunction.Type.values()) {
            assertEquals(0.0f, ActivationFunction.eval(type, 0.0f),
                    "activation=" + type);
        }
    }

    @Test
    void evaluatesSiluForPositiveAndNegativeInputs() {
        assertEquals(0.7310586f, ActivationFunction.eval(ActivationFunction.Type.SILU, 1.0f), 1.0e-6f);
        assertEquals(-0.2689414f, ActivationFunction.eval(ActivationFunction.Type.SILU, -1.0f), 1.0e-6f);
    }

    @Test
    void evaluatesGeluUsingErfApproximation() {
        assertEquals(0.8413447f, ActivationFunction.eval(ActivationFunction.Type.GELU, 1.0f), 1.0e-6f);
        assertEquals(-0.1586553f, ActivationFunction.eval(ActivationFunction.Type.GELU, -1.0f), 1.0e-6f);
    }

    @Test
    void evaluatesPytorchTanhGelu() {
        assertEquals(0.8411920f,
                ActivationFunction.eval(ActivationFunction.Type.GELU_PYTORCH_TANH, 1.0f), 1.0e-6f);
        assertEquals(-0.1588080f,
                ActivationFunction.eval(ActivationFunction.Type.GELU_PYTORCH_TANH, -1.0f), 1.0e-6f);
    }

    @Test
    void evaluatesTanh() {
        assertEquals((float) Math.tanh(1.0), ActivationFunction.eval(ActivationFunction.Type.TANH, 1.0f), 1.0e-6f);
        assertEquals((float) Math.tanh(-1.0), ActivationFunction.eval(ActivationFunction.Type.TANH, -1.0f), 1.0e-6f);
    }
}
