package io.teknek.sketches.guide;

import io.teknek.sketches.SketchesSettings;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class IndexTest {

    @Test
    void basicInterface() {
        Vocabulary vocabulary = new Vocabulary(3, Map.of("1", List.of(1), "2", List.of(2)));
        Index index = new Index("[1-9]", vocabulary);

        int initialState = index.getInitialState();
        assertEquals(0, initialState);
        assertEquals(List.of(1, 2), index.getAllowedTokens(initialState));
        assertEquals(false, index.isFinalState(initialState));

        int nextState = index.getNextState(initialState, 2);
        assertTrue(index.isFinalState(nextState));
        assertEquals(Set.of(nextState), index.getFinalStates());
        assertEquals(List.of(3), index.getAllowedTokens(nextState));
        assertEquals(nextState, index.getNextState(nextState, 3));
    }

    @Test
    void alternativesAllowMultipleTokensAfterSharedPrefix() {
        Vocabulary vocabulary = new Vocabulary(0, Map.of("a", List.of(1), "b", List.of(2), "c", List.of(3)));
        Index index = new Index("a(b|c)", vocabulary);

        int afterA = index.getNextState(index.getInitialState(), 1);

        assertEquals(List.of(2, 3), index.getAllowedTokens(afterA));
        assertTrue(index.isFinalState(index.getNextState(afterA, 2)));
        assertTrue(index.isFinalState(index.getNextState(afterA, 3)));
    }

    @Test
    void multiCharacterTokensCanAdvanceAcrossSeveralRegexCharacters() {
        Vocabulary vocabulary = new Vocabulary(0, Map.of("a", List.of(1), "b", List.of(2), "ab", List.of(4)));
        Index index = new Index("ab", vocabulary);

        assertEquals(List.of(1, 4), index.getAllowedTokens(index.getInitialState()));

        int afterA = index.getNextState(index.getInitialState(), 1);
        assertEquals(List.of(2), index.getAllowedTokens(afterA));

        int afterAb = index.getNextState(index.getInitialState(), 4);
        assertTrue(index.isFinalState(afterAb));
        assertEquals(List.of(0), index.getAllowedTokens(afterAb));
    }

    @Test
    void rejectsMissingTransitions() {
        Vocabulary vocabulary = new Vocabulary(0, Map.of("a", List.of(1), "b", List.of(2)));
        Index index = new Index("a", vocabulary);

        assertThrows(IllegalArgumentException.class, () -> index.getNextState(index.getInitialState(), 2));
    }

    @Test
    void rejectsRegexOverLengthLimit() {
        Vocabulary vocabulary = new Vocabulary(0, Map.of("a", List.of(1)));
        SketchesSettings settings = new SketchesSettings(1, 10, 10);

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> new Index("aa", vocabulary, settings));
        assertTrue(error.getMessage().contains("guided_regex length 2 exceeds maxRegexLength 1"));
    }

    @Test
    void rejectsIndexOverStateLimit() {
        Vocabulary vocabulary = new Vocabulary(0, Map.of("a", List.of(1), "b", List.of(2), "c", List.of(3)));
        SketchesSettings settings = new SketchesSettings(10, 1, 100);

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> new Index("abc", vocabulary, settings));
        assertTrue(error.getMessage().contains("guided_regex index exceeded maxIndexStates 1"));
    }

    @Test
    void rejectsIndexOverTransitionLimit() {
        Vocabulary vocabulary = new Vocabulary(0, Map.of("a", List.of(1), "b", List.of(2), "c", List.of(3)));
        SketchesSettings settings = new SketchesSettings(10, 10, 1);

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> new Index("[abc]", vocabulary, settings));
        assertTrue(error.getMessage().contains("guided_regex index exceeded maxIndexTransitions 1"));
    }
}
