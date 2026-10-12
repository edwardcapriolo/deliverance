package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Objects;

/**
 * EmbeddingGemma 2 RMSNorm operation.
 *
 * <p>The operation follows the Hugging Face contract:
 * {@code output = input / sqrt(mean(input^2) + epsilon) * weight}. A null
 * weight represents the reference implementation's {@code with_scale=false}
 * mode. The generic TensorRef provider helper is used for provider-backed
 * execution; no Gemma 3 or Gemma 4 implementation is reused.</p>
 */
public final class EmbeddingGemma2RmsNormOperation {
    private final float epsilon;

    /**
     * Creates an EmbeddingGemma 2 RMSNorm operation.
     *
     * @param epsilon positive numerical-stability epsilon from the model config
     */
    public EmbeddingGemma2RmsNormOperation(float epsilon) {
        if (!Float.isFinite(epsilon) || epsilon < 0.0f) {
            throw new IllegalArgumentException("epsilon must be finite and non-negative");
        }
        this.epsilon = epsilon;
    }

    /**
     * Applies row-wise RMSNorm to a two-dimensional tensor.
     *
     * @param output writable TensorRef with the same shape as {@code input}
     * @param input TensorRef shaped {@code [rows, hidden]}
     * @param weight optional scale TensorRef shaped {@code [1, hidden]}; null disables scaling
     * @param lighter TensorRef operation/provider owner
     */
    public void apply(TensorRef output, TensorRef input, TensorRef weight, Lighter lighter) {
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(lighter, "lighter");
        new CompositeOps(lighter).rmsNorm(output, input, weight, epsilon, 0.0f);
    }
}
