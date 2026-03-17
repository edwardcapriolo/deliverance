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


    /*
    Based on technical documentation and release information, Gemma 3 utilizes
GeGLU (Gaussian Error Linear Unit-GLU) as its non-linear activation function, replacing standard ReLU non-linearity to enhance model performance.

    Activation Type: Gemma 3 specifically uses gelu_pytorch_tanh within its decoder layers.
    Structure: GeGLU is a variation of the Gated Linear Unit (GLU), which divides the activation into a sigmoidal part and a linear projection, which are then element-wise multiplied.
    Normalization: The model uses RMSNorm (Root Mean Squared Normalization) across various parts of the architecture (input, post-attention, pre-feedforward, and post-feedforward layers) to stabilize training, ensuring that activations do not become too large.

Note: Some investigations into Gemma 3, particularly using the float16 precision, have noted that, in certain edge cases, activations can reach values as high as 800,000, which is significantly higher than the float16 maximum range (65504). Using bfloat16 or upcasting to float32 is recommended.
     */
}