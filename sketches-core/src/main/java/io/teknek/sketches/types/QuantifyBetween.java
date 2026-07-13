package io.teknek.sketches.types;

public class QuantifyBetween extends Term {
    private final Term term;
    private final int min;
    private final int max;

    public QuantifyBetween(Term term, int min, int max) {
        this.term = term;
        this.min = min;
        this.max = max;
    }

    public Term term() {
        return term;
    }

    public int min() {
        return min;
    }

    public int max() {
        return max;
    }
}
