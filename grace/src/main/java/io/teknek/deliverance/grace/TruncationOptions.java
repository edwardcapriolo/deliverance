package io.teknek.deliverance.grace;

public record TruncationOptions(int maxLength, TruncationSide side) {
    public TruncationOptions {
        side = side == null ? TruncationSide.RIGHT : side;
        if (maxLength < 0) {
            throw new IllegalArgumentException("maxLength must be non-negative");
        }
    }

    public static TruncationOptions maxLength(int maxLength) {
        return new TruncationOptions(maxLength, TruncationSide.RIGHT);
    }

    public TruncationOptions withSide(TruncationSide overrideSide) {
        return new TruncationOptions(maxLength, overrideSide);
    }
}
