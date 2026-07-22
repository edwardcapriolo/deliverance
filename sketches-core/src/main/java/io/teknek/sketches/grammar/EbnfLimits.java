package io.teknek.sketches.grammar;

public record EbnfLimits(int maxRepeat, int maxRecursionDepth, int maxAutomatonStates) {
    public static final EbnfLimits DEFAULT = new EbnfLimits(16, 4, 10_000);

    public EbnfLimits {
        if (maxRepeat < 0) {
            throw new IllegalArgumentException("maxRepeat must be >= 0");
        }
        if (maxRecursionDepth < 0) {
            throw new IllegalArgumentException("maxRecursionDepth must be >= 0");
        }
        if (maxAutomatonStates < 1) {
            throw new IllegalArgumentException("maxAutomatonStates must be >= 1");
        }
    }
}
