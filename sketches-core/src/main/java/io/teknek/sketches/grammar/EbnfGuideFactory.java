package io.teknek.sketches.grammar;

import dk.brics.automaton.Automaton;
import io.teknek.sketches.SketchesSettings;
import io.teknek.sketches.guide.Index;
import io.teknek.sketches.guide.IndexGuide;
import io.teknek.sketches.guide.Vocabulary;

import java.util.Objects;

public final class EbnfGuideFactory {
    private EbnfGuideFactory() {
    }

    public static IndexGuide create(String definition, String startRule, Vocabulary vocabulary) {
        return create(definition, startRule, vocabulary, EbnfLimits.DEFAULT, SketchesSettings.DEFAULT);
    }

    public static IndexGuide create(String definition, String startRule, Vocabulary vocabulary, EbnfLimits ebnfLimits,
            SketchesSettings indexSettings) {
        Objects.requireNonNull(vocabulary, "vocabulary");
        Automaton automaton = EbnfCompiler.compile(definition, startRule, ebnfLimits);
        return new IndexGuide(new Index(automaton, vocabulary, indexSettings));
    }
}
