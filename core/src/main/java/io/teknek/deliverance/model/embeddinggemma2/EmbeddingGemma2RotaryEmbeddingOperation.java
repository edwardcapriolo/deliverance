package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Objects;

/** Generates and applies default RoPE for one EmbeddingGemma 2 attention layer type. */
public final class EmbeddingGemma2RotaryEmbeddingOperation {
    private final int headSize;
    private final int heads;
    private final float[][] frequencies;

    /**
     * Creates default RoPE frequencies for positions beginning at zero.
     *
     * @param headSize even per-head rotary width
     * @param heads number of heads to rotate
     * @param ropeTheta RoPE base from the layer type configuration
     * @param positions number of positions covered by the frequency table
     */
    public EmbeddingGemma2RotaryEmbeddingOperation(int headSize, int heads, double ropeTheta, int positions) {
        if (headSize <= 0 || (headSize & 1) != 0 || heads <= 0 || !Double.isFinite(ropeTheta)
                || ropeTheta <= 0.0 || positions <= 0) {
            throw new IllegalArgumentException("Invalid EmbeddingGemma 2 RoPE configuration");
        }
        this.headSize = headSize;
        this.heads = heads;
        int half = headSize / 2;
        this.frequencies = new float[positions * half][2];
        for (int position = 0; position < positions; position++) {
            for (int dimension = 0; dimension < half; dimension++) {
                double inverse = 1.0 / Math.pow(ropeTheta, (2.0 * dimension) / headSize);
                double angle = position * inverse;
                frequencies[position * half + dimension][0] = (float) Math.cos(angle);
                frequencies[position * half + dimension][1] = (float) Math.sin(angle);
            }
        }
    }

    /** Applies the configured rotation in place to all target rows. */
    public void apply(TensorRef target, int startPosition, Lighter lighter) {
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(lighter, "lighter");
        new CompositeOps(lighter).rotaryEmbedding(target, heads, headSize, startPosition, frequencies);
    }
}
