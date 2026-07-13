package io.teknek.sketches.types;

public class KleenePlus extends Term {
    private final Term term;

    public KleenePlus(Term term) {
        this.term = term;
    }

    public Term term() {
        return term;
    }
}
