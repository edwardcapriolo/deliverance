package io.teknek.sketches.guide;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class VocabularyTest {

    @Test
    void basicVocabularyInterface() {
        Vocabulary vocabulary = vocabulary();

        assertEquals(3, vocabulary.getEosTokenId());
        assertEquals(List.of(3), vocabulary.getEosTokenIds());
        assertEquals(List.of(1), vocabulary.get("1"));
        assertEquals(3, vocabulary.size());

        vocabulary.insert("b", 4);
        assertEquals(List.of(4), vocabulary.get("b"));
        assertEquals("b", vocabulary.tokenText(4));
        assertEquals(4, vocabulary.size());

        vocabulary.insert("b", 5);
        assertEquals(List.of(4, 5), vocabulary.get("b"));
        assertEquals("b", vocabulary.tokenText(5));
        assertEquals(5, vocabulary.size());

        vocabulary.remove("b");
        assertNull(vocabulary.get("b"));
        assertNull(vocabulary.tokenText(4));
        assertNull(vocabulary.tokenText(5));

        vocabulary.remove("b");
        assertNull(vocabulary.get("b"));

        assertEquals(List.of(2), vocabulary.get("a"));
        vocabulary.remove("a");
        assertNull(vocabulary.get("a"));
    }

    @Test
    void stringTokensProduceEquivalentVocabularies() {
        Vocabulary vocabulary = new Vocabulary(3, Map.of("1", List.of(1), "a", List.of(2)));
        Vocabulary same = new Vocabulary(3, Map.of("1", List.of(1), "a", List.of(2)));

        assertEquals(vocabulary.getEosTokenId(), same.getEosTokenId());
        assertEquals(vocabulary.get("1"), same.get("1"));
        assertEquals(vocabulary.size(), same.size());
        assertEquals(vocabulary, same);
    }

    @Test
    void rejectsBadInputs() {
        assertThrows(NullPointerException.class, () -> new Vocabulary(0, null));
        assertThrows(IllegalArgumentException.class, () -> new Vocabulary(0, java.util.Collections.singletonMap(null, List.of(1))));
        assertThrows(NullPointerException.class, () -> new Vocabulary(0, Map.of("a", null)));

        Vocabulary vocabulary = vocabulary();
        assertThrows(IllegalArgumentException.class, () -> vocabulary.get(null));
        assertThrows(IllegalArgumentException.class, () -> vocabulary.insert(null, 6));
    }

    @Test
    void rejectsEosTokenInsertion() {
        Vocabulary vocabulary = vocabulary();
        assertThrows(IllegalArgumentException.class, () -> vocabulary.insert("eos-token", 3));
    }

    @Test
    void supportsMultipleEosTokens() {
        Vocabulary vocabulary = new Vocabulary(List.of(3, 4), Map.of("1", List.of(1), "a", List.of(2)));

        assertEquals(3, vocabulary.getEosTokenId());
        assertEquals(List.of(3, 4), vocabulary.getEosTokenIds());
        assertEquals(4, vocabulary.size());
        assertThrows(IllegalArgumentException.class, () -> vocabulary.insert("eos-token", 4));
    }

    @Test
    void copyAndEquality() {
        Vocabulary vocabulary = vocabulary();
        Vocabulary vocabulary2 = new Vocabulary(vocabulary);
        assertEquals(vocabulary, vocabulary2);

        Vocabulary copyVocabulary2 = new Vocabulary(vocabulary2);
        assertEquals(copyVocabulary2, vocabulary2);

        vocabulary2.insert("new", 4);
        assertNotEquals(vocabulary2, copyVocabulary2);
        assertEquals(vocabulary2.size() - 1, copyVocabulary2.size());
        assertEquals(copyVocabulary2, vocabulary);
        assertEquals(copyVocabulary2.size(), vocabulary.size());
    }

    private Vocabulary vocabulary() {
        return new Vocabulary(3, Map.of("1", List.of(1), "a", List.of(2)));
    }
}
