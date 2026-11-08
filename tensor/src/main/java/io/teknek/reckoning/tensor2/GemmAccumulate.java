package io.teknek.reckoning.tensor2;

import java.util.Objects;

/** Computes {@code target += scale * (left * right)} for dense rank-2 tensors. */
public final class GemmAccumulate {
    private final TensorRef target;
    private final TensorRef left;
    private final TensorRef right;
    private final float scale;

    public GemmAccumulate(TensorRef target, TensorRef left, TensorRef right, float scale) {
        this.target = Objects.requireNonNull(target, "target");
        this.left = Objects.requireNonNull(left, "left");
        this.right = Objects.requireNonNull(right, "right");
        this.scale = scale;
    }

    public TensorRef target() { return target; }
    public TensorRef left() { return left; }
    public TensorRef right() { return right; }
    public float scale() { return scale; }
}
