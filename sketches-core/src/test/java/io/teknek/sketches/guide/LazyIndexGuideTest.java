package io.teknek.sketches.guide;

import io.teknek.sketches.json.JsonSchemaRegexBuilder;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LazyIndexGuideTest {

    @Test
    void computesTransitionsOnlyForQueriedStates() {
        Vocabulary vocabulary = new Vocabulary(3, Map.of("1", List.of(1), "2", List.of(2)));
        LazyIndex index = new LazyIndex("[1-9]", vocabulary);
        LazyIndexGuide guide = new LazyIndexGuide(index);

        assertEquals(0, index.computedStateCount());
        assertEquals(List.of(1, 2), guide.getTokens());
        assertEquals(1, index.computedStateCount());
        assertEquals(List.of(3), guide.advance(1));
        assertEquals(2, index.computedStateCount());
        assertTrue(guide.isFinished());
    }

    @Test
    void objectKeyStateDoesNotAllowInvalidWhitespaceToken() {
        String schema = """
                {
                  "type": "object",
                  "properties": {
                    "places": {
                      "type": "array",
                      "items": { "type": "string" }
                    }
                  }
                }
                """;
        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(schema);
        Vocabulary vocabulary = new Vocabulary(99, Map.of(
                "{", List.of(1),
                "\"places\"", List.of(2),
                " ", List.of(3),
                " n", List.of(4),
                ":", List.of(5),
                "[]", List.of(6),
                "}", List.of(7)));
        LazyIndexGuide guide = new LazyIndexGuide(new LazyIndex(regex, vocabulary));

        guide.advance(1);
        guide.advance(2);

        assertTrue(guide.getTokens().contains(5), "colon should be valid after object key");
        assertFalse(guide.getTokens().contains(3), "default guided JSON should use tight structural binding");
        assertFalse(guide.getTokens().contains(4), "token ' n' would produce invalid JSON after a completed key");
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("jsonBoundaryPrefixes")
    void lazyGuideAllowedTokensMatchEagerGuideAtJsonBoundaries(String label, List<Integer> acceptedTokens) {
        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(itemsSchema());
        Vocabulary vocabulary = jsonBoundaryVocabulary();
        IndexGuide eager = new IndexGuide(new Index(regex, vocabulary));
        LazyIndexGuide lazy = new LazyIndexGuide(new LazyIndex(regex, vocabulary));

        for (Integer token : acceptedTokens) {
            eager.advance(token);
            lazy.advance(token);
        }

        assertEquals(sorted(eager.getTokens()), sorted(lazy.getTokens()), label);
    }

    @Test
    void rawNewlineTokenIsNotAllowedInsideJsonString() {
        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(itemsSchema());
        Vocabulary vocabulary = jsonBoundaryVocabulary();
        LazyIndexGuide guide = new LazyIndexGuide(new LazyIndex(regex, vocabulary));

        guide.advance(1); // {
        guide.advance(2); // "items"
        guide.advance(3); // :
        guide.advance(4); // [

        assertFalse(guide.getTokens().contains(15), "raw newline inside JSON string must not be allowed");
    }

    @Test
    void lazyGuideMatchesEagerGuideThroughCompleteJsonDocument() {
        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(itemsSchema());
        Vocabulary vocabulary = jsonBoundaryVocabulary();
        IndexGuide eager = new IndexGuide(new Index(regex, vocabulary));
        LazyIndexGuide lazy = new LazyIndexGuide(new LazyIndex(regex, vocabulary));
        List<Integer> document = List.of(1, 2, 3, 4, 5, 13, 8, 9);

        for (Integer token : document) {
            assertEquals(sorted(eager.getTokens()), sorted(lazy.getTokens()), "before token " + token);
            eager.advance(token);
            lazy.advance(token);
        }

        assertEquals(sorted(eager.getTokens()), sorted(lazy.getTokens()), "after complete document");
        assertTrue(eager.isFinished());
        assertTrue(lazy.isFinished());
        assertEquals(List.of(99), lazy.getTokens());
    }

    @Test
    void structuralCharactersAreAllowedInsideJsonStringOnlyWhenQuoted() {
        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(itemsSchemaAtLeastOneItem());
        Vocabulary vocabulary = jsonBoundaryVocabulary();
        LazyIndexGuide guide = new LazyIndexGuide(new LazyIndex(regex, vocabulary));

        guide.advance(1); // {
        guide.advance(2); // "items"
        guide.advance(3); // :
        guide.advance(4); // [

        assertTrue(guide.getTokens().contains(16), "quoted right-brace string should be valid string content");
        assertTrue(guide.getTokens().contains(17), "quoted right-bracket string should be valid string content");
        assertFalse(guide.getTokens().contains(9), "raw object close is not valid as an array item");
        assertFalse(guide.getTokens().contains(8), "raw array close is not valid before an item in this schema path");
    }

    private static Stream<Arguments> jsonBoundaryPrefixes() {
        return Stream.of(
                Arguments.of("initial", List.of()),
                Arguments.of("after object open", List.of(1)),
                Arguments.of("after object key", List.of(1, 2)),
                Arguments.of("after colon", List.of(1, 2, 3)),
                Arguments.of("after array open", List.of(1, 2, 3, 4)),
                Arguments.of("after first string item", List.of(1, 2, 3, 4, 5)),
                Arguments.of("after array comma", List.of(1, 2, 3, 4, 5, 6)),
                Arguments.of("after compact comma-plus-string token", List.of(1, 2, 3, 4, 5, 13)),
                Arguments.of("after array close", List.of(1, 2, 3, 4, 5, 8))
        );
    }

    private static String itemsSchema() {
        return """
                {
                  "type": "object",
                  "properties": {
                    "items": {
                      "type": "array",
                      "items": { "type": "string" }
                    }
                  }
                }
                """;
    }

    private static String itemsSchemaAtLeastOneItem() {
        return """
                {
                  "type": "object",
                  "properties": {
                    "items": {
                      "type": "array",
                      "minItems": 1,
                      "maxItems": 3,
                      "items": { "type": "string" }
                    }
                  }
                }
                """;
    }

    private static Vocabulary jsonBoundaryVocabulary() {
        return new Vocabulary(99, Map.ofEntries(
                Map.entry("{", List.of(1)),
                Map.entry("\"items\"", List.of(2)),
                Map.entry(":", List.of(3)),
                Map.entry("[", List.of(4)),
                Map.entry("\"trophy\"", List.of(5)),
                Map.entry(",", List.of(6)),
                Map.entry("\"ledger\"", List.of(7)),
                Map.entry("]", List.of(8)),
                Map.entry("}", List.of(9)),
                Map.entry(" ", List.of(10)),
                Map.entry(" n", List.of(11)),
                Map.entry(",ledger", List.of(12)),
                Map.entry(",\"ledger\"", List.of(13)),
                Map.entry("\n", List.of(14)),
                Map.entry("\"bad\nvalue\"", List.of(15)),
                Map.entry("\"}\"", List.of(16)),
                Map.entry("\"]\"", List.of(17))));
    }

    private static List<Integer> sorted(List<Integer> values) {
        List<Integer> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        return sorted;
    }
}
