package io.teknek.sketches.types;

public class KleeneStar extends Term {
    private final Term term;

    public KleeneStar(Term term) {
        this.term = term;
    }

    public Term term() {
        return term;
    }
}
