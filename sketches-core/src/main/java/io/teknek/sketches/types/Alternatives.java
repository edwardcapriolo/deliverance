package io.teknek.sketches.types;

import java.util.List;

public class Alternatives extends Term {
    private final List<Term> terms;

    public Alternatives(List<Term> terms) {
        this.terms = List.copyOf(terms);
    }

    public List<Term> terms() {
        return terms;
    }
}
