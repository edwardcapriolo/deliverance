package io.teknek.sketches.guide;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class OutlinesCoreBackendVocabularyPortTest {

    @Test
    void outlinesTestCreateVocabularyPreservesDuplicateTokenIdsPort() {
        // Upstream: outlines/tests/backends/test_outlines_core.py::test_create_vocabulary_preserves_duplicate_token_ids
        Map<String, Integer> vocab = new LinkedHashMap<>();
        vocab.put("hello", 1);
        vocab.put("world", 2);
        vocab.put("<0x20>", 3);
        vocab.put("▁", 4);

        Vocabulary vocabulary = createOutlinesCoreVocabulary(vocab, 0, "hello", token -> {
            if (token.equals("<0x20>") || token.equals("▁")) {
                return " ";
            }
            return token;
        });

        assertEquals(4, vocabulary.size());
    }

    @Test
    void outlinesTestCreateVocabularyPreservesDistinctDecodedStringsPort() {
        // Upstream: outlines/tests/backends/test_outlines_core.py::test_create_vocabulary_preserves_distinct_decoded_strings
        Map<String, Integer> vocab = new LinkedHashMap<>();
        vocab.put("▁hello", 1);
        vocab.put("hello", 2);
        vocab.put("▁the", 3);
        vocab.put("<eos>", 0);

        Vocabulary vocabulary = createOutlinesCoreVocabulary(vocab, 0, "<eos>",
                token -> token.startsWith("▁") ? token.replace("▁", " ") : token);

        assertEquals(List.of(1), vocabulary.get(" hello"));
        assertEquals(List.of(2), vocabulary.get("hello"));
        assertEquals(List.of(3), vocabulary.get(" the"));
        assertFalse(vocabulary.tokens().containsKey("<eos>"));
        assertEquals(4, vocabulary.size());
        assertEquals(3, vocabulary.tokens().values().stream().mapToInt(List::size).sum());
    }

    @Test
    void outlinesTestCreateVocabularyDuplicateDecodedStringsPort() {
        // Upstream: outlines/tests/backends/test_outlines_core.py::test_create_vocabulary_duplicate_decoded_strings
        Map<String, Integer> vocab = new LinkedHashMap<>();
        vocab.put("▁hi", 1);
        vocab.put(" hi", 2);
        vocab.put("hi", 3);
        vocab.put("<eos>", 0);

        Vocabulary vocabulary = createOutlinesCoreVocabulary(vocab, 0, "<eos>",
                token -> token.startsWith("▁") ? token.replace("▁", " ") : token);

        assertEquals(List.of(1, 2), vocabulary.get(" hi"));
        assertEquals(List.of(3), vocabulary.get("hi"));
        assertEquals(4, vocabulary.size());
        assertEquals(3, vocabulary.tokens().values().stream().mapToInt(List::size).sum());
    }

    private static Vocabulary createOutlinesCoreVocabulary(Map<String, Integer> vocab, int eosTokenId, String eosToken,
            Function<String, String> tokenToString) {
        Vocabulary vocabulary = new Vocabulary(eosTokenId, Map.of());
        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            if (entry.getKey().equals(eosToken) || entry.getValue() == eosTokenId) {
                continue;
            }
            vocabulary.insert(tokenToString.apply(entry.getKey()), entry.getValue());
        }
        return vocabulary;
    }
}
