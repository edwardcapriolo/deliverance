package io.teknek.deliverance.math;

import net.jafama.FastMath;
public class ActivationFunction {

    public enum Type {
        SILU,
        GELU,
        TANH,
        GELU_PYTORCH_TANH
    }

    public static float eval(Type t, float x) {
        return switch (t) {
            case SILU -> (float) (x * (1.0f / (1.0f + FastMath.exp(-x))));
            case GELU, GELU_PYTORCH_TANH -> (float) (0.5 * x * (1 + FastMath.tanh(
                    FastMath.sqrt(2 / Math.PI) * (x + 0.044715 * FastMath.pow(x, 3))
            )));
            case TANH -> (float) FastMath.tanh(x);
        };
    }
}