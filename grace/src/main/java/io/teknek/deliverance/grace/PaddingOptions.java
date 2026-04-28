package io.teknek.deliverance.grace;

public record PaddingOptions(PaddingStrategy strategy, Integer maxLength, Integer padToMultipleOf, PaddingSide side) {
    public PaddingOptions {
        strategy = strategy == null ? PaddingStrategy.DO_NOT_PAD : strategy;
        if (maxLength != null && maxLength < 0) {
            throw new IllegalArgumentException("maxLength must be non-negative");
        }
        if (padToMultipleOf != null && padToMultipleOf <= 0) {
            throw new IllegalArgumentException("padToMultipleOf must be positive");
        }
    }

    public static PaddingOptions doNotPad() {
        return new PaddingOptions(PaddingStrategy.DO_NOT_PAD, null, null, null);
    }

    public static PaddingOptions longest() {
        return new PaddingOptions(PaddingStrategy.LONGEST, null, null, null);
    }

    public static PaddingOptions maxLength(int maxLength) {
        return new PaddingOptions(PaddingStrategy.MAX_LENGTH, maxLength, null, null);
    }

    public PaddingOptions withPadToMultipleOf(int multiple) {
        return new PaddingOptions(strategy, maxLength, multiple, side);
    }

    public PaddingOptions withSide(PaddingSide overrideSide) {
        return new PaddingOptions(strategy, maxLength, padToMultipleOf, overrideSide);
    }
}
