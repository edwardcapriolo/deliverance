package io.teknek.sketches.grammar;

import dk.brics.automaton.Automaton;
import dk.brics.automaton.BasicAutomata;
import dk.brics.automaton.BasicOperations;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Compiles bounded Deliverance EBNF v1 grammars to finite automata. */
public final class EbnfCompiler {
    private final EbnfGrammar grammar;
    private final EbnfLimits limits;

    public EbnfCompiler(EbnfGrammar grammar, EbnfLimits limits) {
        this.grammar = Objects.requireNonNull(grammar, "grammar");
        this.limits = Objects.requireNonNull(limits, "limits");
    }

    public static Automaton compile(String definition, String startRule) {
        return compile(definition, startRule, EbnfLimits.DEFAULT);
    }

    public static Automaton compile(String definition, String startRule, EbnfLimits limits) {
        EbnfGrammar grammar = EbnfParser.parse(definition);
        return new EbnfCompiler(grammar, limits).compile(startRule);
    }

    public Automaton compile(String startRule) {
        String start = startRule == null || startRule.isBlank() ? "root" : startRule;
        Automaton automaton = compileNode(grammar.rule(start), new HashMap<>());
        automaton.determinize();
        automaton.minimize();
        if (automaton.getNumberOfStates() > limits.maxAutomatonStates()) {
            throw new IllegalArgumentException("guided_grammar automaton exceeded maxAutomatonStates "
                    + limits.maxAutomatonStates());
        }
        return automaton;
    }

    private Automaton compileNode(EbnfNode node, Map<String, Integer> recursionCounts) {
        return switch (node) {
            case EbnfNode.Literal literal -> BasicAutomata.makeString(literal.value());
            case EbnfNode.Ref ref -> compileRef(ref, recursionCounts);
            case EbnfNode.Seq seq -> compileSeq(seq, recursionCounts);
            case EbnfNode.Alt alt -> compileAlt(alt, recursionCounts);
            case EbnfNode.Repeat repeat -> compileRepeat(repeat, recursionCounts);
        };
    }

    private Automaton compileRef(EbnfNode.Ref ref, Map<String, Integer> recursionCounts) {
        int count = recursionCounts.getOrDefault(ref.name(), 0);
        if (count >= limits.maxRecursionDepth()) {
            return BasicAutomata.makeEmpty();
        }
        recursionCounts.put(ref.name(), count + 1);
        try {
            return compileNode(grammar.rule(ref.name()), recursionCounts);
        } finally {
            if (count == 0) {
                recursionCounts.remove(ref.name());
            } else {
                recursionCounts.put(ref.name(), count);
            }
        }
    }

    private Automaton compileSeq(EbnfNode.Seq seq, Map<String, Integer> recursionCounts) {
        List<Automaton> parts = new ArrayList<>();
        for (EbnfNode part : seq.parts()) {
            Automaton compiled = compileNode(part, recursionCounts);
            if (compiled.isEmpty()) {
                return BasicAutomata.makeEmpty();
            }
            parts.add(compiled);
        }
        return BasicOperations.concatenate(parts);
    }

    private Automaton compileAlt(EbnfNode.Alt alt, Map<String, Integer> recursionCounts) {
        List<Automaton> options = new ArrayList<>();
        for (EbnfNode option : alt.options()) {
            Automaton compiled = compileNode(option, recursionCounts);
            if (!compiled.isEmpty()) {
                options.add(compiled);
            }
        }
        if (options.isEmpty()) {
            return BasicAutomata.makeEmpty();
        }
        return BasicOperations.union(options);
    }

    private Automaton compileRepeat(EbnfNode.Repeat repeat, Map<String, Integer> recursionCounts) {
        Automaton compiled = compileNode(repeat.node(), recursionCounts);
        if (compiled.isEmpty()) {
            return repeat.min() == 0 ? BasicAutomata.makeEmptyString() : BasicAutomata.makeEmpty();
        }
        int max = repeat.max() == EbnfNode.Repeat.UNBOUNDED ? limits.maxRepeat() : repeat.max();
        if (max < repeat.min()) {
            return BasicAutomata.makeEmpty();
        }
        return BasicOperations.repeat(compiled, repeat.min(), max);
    }
}
