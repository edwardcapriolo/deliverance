package io.teknek.sketches.types;

public class Optional extends Term {
    private final Term term;

    public Optional(Term term) {
        this.term = term;
    }

    public Term term() {
        return term;
    }
}
