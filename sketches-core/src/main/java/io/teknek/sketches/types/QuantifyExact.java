package io.teknek.sketches.types;

public class QuantifyExact extends Term {
    private final Term term;
    private final int count;

    public QuantifyExact(Term term, int count) {
        this.term = term;
        this.count = count;
    }

    public Term term() {
        return term;
    }

    public int count() {
        return count;
    }
}
