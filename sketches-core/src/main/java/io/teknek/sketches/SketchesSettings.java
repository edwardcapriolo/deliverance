package io.teknek.sketches;

/**
 * Server-side safety limits for sketches guide construction.
 *
 * <p>These limits are operator/model-load settings, not end-user request parameters. They protect the service from regexes
 * that are too large or compile into too many token transitions.</p>
 */
public record SketchesSettings(int maxRegexLength, int maxIndexStates, int maxIndexTransitions,
                               GuidedIndexMode guidedIndexMode) {
    public enum GuidedIndexMode {
        EAGER,
        LAZY
    }

    public static final SketchesSettings DEFAULT = new SketchesSettings(10_000, 10_000, 2_000_000,
            GuidedIndexMode.LAZY);

    public SketchesSettings(int maxRegexLength, int maxIndexStates, int maxIndexTransitions) {
        this(maxRegexLength, maxIndexStates, maxIndexTransitions, GuidedIndexMode.LAZY);
    }

    public SketchesSettings {
        if (guidedIndexMode == null) {
            throw new IllegalArgumentException("guidedIndexMode must not be null");
        }
        if (maxRegexLength < 1) {
            throw new IllegalArgumentException("maxRegexLength must be >= 1");
        }
        if (maxIndexStates < 1) {
            throw new IllegalArgumentException("maxIndexStates must be >= 1");
        }
        if (maxIndexTransitions < 1) {
            throw new IllegalArgumentException("maxIndexTransitions must be >= 1");
        }
    }
}
