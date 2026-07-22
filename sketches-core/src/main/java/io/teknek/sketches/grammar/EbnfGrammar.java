package io.teknek.sketches.grammar;

import java.util.Map;

public record EbnfGrammar(Map<String, EbnfNode> rules) {
    public EbnfGrammar {
        rules = Map.copyOf(rules);
    }

    public EbnfNode rule(String name) {
        EbnfNode rule = rules.get(name);
        if (rule == null) {
            throw new IllegalArgumentException("unknown EBNF rule: " + name);
        }
        return rule;
    }
}
