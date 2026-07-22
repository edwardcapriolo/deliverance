package io.teknek.sketches.grammar;

import dk.brics.automaton.Automaton;

public final class EbnfMatcher {
    private final Automaton automaton;

    public EbnfMatcher(String definition, String startRule) {
        this(definition, startRule, EbnfLimits.DEFAULT);
    }

    public EbnfMatcher(String definition, String startRule, EbnfLimits limits) {
        this.automaton = EbnfCompiler.compile(definition, startRule, limits);
    }

    public boolean matches(String value) {
        return automaton.run(value);
    }

    public int stateCount() {
        return automaton.getNumberOfStates();
    }
}
