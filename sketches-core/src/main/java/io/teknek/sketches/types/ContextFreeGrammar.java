package io.teknek.sketches.types;

public class ContextFreeGrammar extends Term {
    private final String definition;

    public ContextFreeGrammar(String definition) {
        this.definition = definition;
    }

    public String definition() {
        return definition;
    }
}
