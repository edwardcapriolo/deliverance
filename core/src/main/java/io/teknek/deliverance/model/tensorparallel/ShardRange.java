package io.teknek.deliverance.model.tensorparallel;

public record ShardRange(int startInclusive, int endExclusive) {
    public ShardRange {
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
