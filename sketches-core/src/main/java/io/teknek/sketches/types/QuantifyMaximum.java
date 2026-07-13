package io.teknek.sketches.types;

public class QuantifyMaximum extends Term {
    private final Term term;
    private final int max;

    public QuantifyMaximum(Term term, int max) {
        this.term = term;
        this.max = max;
    }

    public Term term() {
        return term;
    }

    public int max() {
        return max;
    }
}
