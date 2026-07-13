package io.teknek.sketches.guide;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class IndexGuideTest {

    @Test
    void interfaceMatchesRegexFsaAllowedTokenFlow() {
        Vocabulary vocabulary = new Vocabulary(3, Map.of("1", List.of(1), "a", List.of(2)));
        Index index = new Index("[1-9]", vocabulary);
        IndexGuide guide = new IndexGuide(index);

        assertEquals(index.getInitialState(), guide.getState());
        assertEquals(List.of(1), guide.getTokens());

        assertEquals(List.of(3), guide.advance(1));
        assertTrue(guide.isFinished());
        assertEquals(List.of(3), guide.getTokens());

        assertEquals(List.of(3), guide.advance(3));
        assertTrue(guide.isFinished());
        assertThrows(IllegalArgumentException.class, () -> guide.advance(4));
    }

    @Test
    void walksRegexFinalStateThroughOptionalBranch() {
        Vocabulary vocabulary = new Vocabulary(104, Map.of("\n", List.of(103), ".",
                List.of(102), "`", List.of(101)));
        Index index = new Index("`\n(\\.\n)?`\n", vocabulary);
        IndexGuide guide = new IndexGuide(index);

        assertEquals(List.of(101), guide.getTokens());
        assertEquals(List.of(103), guide.advance(101));
        assertEquals(List.of(101, 102), guide.advance(103));
        assertEquals(List.of(103), guide.advance(101));
        assertEquals(List.of(104), guide.advance(103));
        assertTrue(guide.isFinished());
    }

    @Test
    void tokenTransitionsForEquivalentAlternativesAreIdentical() {
        Vocabulary vocabulary = new Vocabulary(4, Map.of("a", List.of(1), "b", List.of(2), "z", List.of(3)));
        Index index = new Index("z[ab]z", vocabulary);

        IndexGuide guide1 = new IndexGuide(index);
        IndexGuide guide2 = new IndexGuide(index);

        assertEquals(guide1.advance(3), guide2.advance(3));
        assertEquals(guide1.advance(1), guide2.advance(2));
        assertEquals(List.of(4), guide1.advance(3));
        assertEquals(List.of(4), guide2.advance(3));
        assertTrue(guide1.isFinished());
        assertTrue(guide2.isFinished());
    }

    @Test
    void acceptsTokensReportsCompleteRegexMatches() {
        Vocabulary vocabulary = new Vocabulary(3, Map.of("1", List.of(1), "2", List.of(2)));
        IndexGuide guide = new IndexGuide(new Index("[1-9]", vocabulary));

        assertTrue(guide.acceptsTokens(List.of(1)));
        assertTrue(guide.acceptsTokens(List.of(2)));
        assertFalse(guide.acceptsTokens(List.of(1, 1)));
        assertFalse(guide.acceptsTokens(List.of(2, 3)));
    }
}
