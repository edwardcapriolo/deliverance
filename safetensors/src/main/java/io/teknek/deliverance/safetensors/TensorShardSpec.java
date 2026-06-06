package io.teknek.deliverance.safetensors;

public record TensorShardSpec(TensorShardAxis axis, int startInclusive, int endExclusive) {
    public TensorShardSpec {
        if (axis == null) {
            throw new IllegalArgumentException("axis must not be null");
        }
        if (startInclusive < 0) {
            throw new IllegalArgumentException("startInclusive must be >= 0");
        }
        if (endExclusive < startInclusive) {
            throw new IllegalArgumentException("endExclusive must be >= startInclusive");
        }
    }

    public int length() {
        return endExclusive - startInclusive;
    }
}
