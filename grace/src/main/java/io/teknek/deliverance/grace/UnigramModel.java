package io.teknek.deliverance.grace;

import java.util.LinkedHashMap;
import java.util.Map;

/** Vocabulary and scores for a Hugging Face tokenizers Unigram model. */
public record UnigramModel(Map<String, Integer> vocab, Map<String, Float> scores, int unknownId) {
    public UnigramModel {
        vocab = Map.copyOf(new LinkedHashMap<>(vocab));
        scores = Map.copyOf(new LinkedHashMap<>(scores));
    }
}
