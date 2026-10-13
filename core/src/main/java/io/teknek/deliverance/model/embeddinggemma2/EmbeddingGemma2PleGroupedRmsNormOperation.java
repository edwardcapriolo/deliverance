package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Objects;

/** Applies independent RMSNorm to the contiguous per-layer PLE slices. */
public final class EmbeddingGemma2PleGroupedRmsNormOperation {
    private final int groups;
    private final int groupSize;
    private final float epsilon;

    /**
     * Creates grouped PLE normalization without reshaping or copying the projection output.
     *
     * @param groups number of transformer-layer PLE slices
     * @param groupSize width of each PLE slice
     * @param epsilon RMSNorm epsilon
     */
    public EmbeddingGemma2PleGroupedRmsNormOperation(int groups, int groupSize, float epsilon) {
        if (groups <= 0 || groupSize <= 0 || !Float.isFinite(epsilon) || epsilon < 0.0f) {
            throw new IllegalArgumentException("Invalid grouped PLE RMSNorm configuration");
        }
        this.groups = groups;
        this.groupSize = groupSize;
        this.epsilon = epsilon;
    }

    /**
     * Normalizes every layer slice independently in one TensorRef operation.
     *
     * @param target projected PLE tensor shaped {@code [tokens, groups * groupSize]}
     * @param weights optional learned scale shaped {@code [1, groupSize]}
     * @param lighter TensorRef operation/provider owner
     */
    public void apply(TensorRef target, TensorRef weights, Lighter lighter) {
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(lighter, "lighter");
        new CompositeOps(lighter).groupedRmsNormInPlace(target, groups, groupSize, epsilon, weights);
    }
}
