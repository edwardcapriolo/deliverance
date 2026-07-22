package io.teknek.sketches.grammar;

import io.teknek.sketches.guide.IndexGuide;
import io.teknek.sketches.guide.Vocabulary;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EbnfGuideFactoryTest {
    @Test
    void projectsEbnfAutomatonOntoVocabularyTokens() {
        Vocabulary vocabulary = new Vocabulary(99, Map.of(
                "SELECT ", List.of(1),
                "col_1 ", List.of(2),
                "col_2 ", List.of(3),
                " from ", List.of(4),
                "table_1", List.of(5),
                "table_2", List.of(6),
                "bad", List.of(7)));
        IndexGuide guide = EbnfGuideFactory.create("""
                root ::= "SELECT " column " from " table
                column ::= "col_1 " | "col_2 "
                table ::= "table_1" | "table_2"
                """, "root", vocabulary);

        assertEquals(List.of(1), guide.getTokens());
        assertEquals(List.of(2, 3), guide.advance(1));
        assertEquals(List.of(4), guide.advance(3));
        assertEquals(List.of(5, 6), guide.advance(4));
        assertEquals(List.of(99), guide.advance(6));
        assertTrue(guide.isFinished());
    }

    @Test
    void supportsTokensThatCrossLiteralBoundaries() {
        Vocabulary vocabulary = new Vocabulary(0, Map.of(
                "ab", List.of(1),
                "a", List.of(2),
                "b", List.of(3)));
        IndexGuide guide = EbnfGuideFactory.create("root ::= " + quote("a") + " " + quote("b"), "root", vocabulary);

        assertEquals(List.of(1, 2), guide.getTokens());
        assertEquals(List.of(0), guide.advance(1));
        assertTrue(guide.isFinished());
    }

    @Test
    void eosIsOnlyAllowedAtAcceptingState() {
        Vocabulary vocabulary = new Vocabulary(0, Map.of("a", List.of(1), "b", List.of(2)));
        IndexGuide guide = EbnfGuideFactory.create("root ::= " + quote("a") + " " + quote("b"), "root", vocabulary);

        assertFalse(guide.getTokens().contains(0));
        assertEquals(List.of(2), guide.advance(1));
        assertFalse(guide.getTokens().contains(0));
        assertEquals(List.of(0), guide.advance(2));
        assertTrue(guide.isFinished());
    }

    @Test
    void rejectsMissingStartRuleInGuideFactory() {
        Vocabulary vocabulary = new Vocabulary(0, Map.of("x", List.of(1)));

        assertThrows(IllegalArgumentException.class,
                () -> EbnfGuideFactory.create("document ::= " + quote("x"), "root", vocabulary));
    }

    private static String quote(String value) {
        return "\"" + value + "\"";
    }
}
