package io.teknek.deliverance.grace;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public record BytePairEncodingModel(Map<String, Integer> vocab,
                                    Map<String, Integer> mergeRanks,
                                    PreTokenizerConfig preTokenizer,
                                    String unkToken) {
    public BytePairEncodingModel {
        vocab = Map.copyOf(new LinkedHashMap<>(vocab));
        mergeRanks = Map.copyOf(new LinkedHashMap<>(mergeRanks));
        preTokenizer = preTokenizer == null ? new PreTokenizerConfig(null, false, false) : preTokenizer;
    }

    public static Map<String, Integer> fromMerges(List<String> merges) {
        Map<String, Integer> ranks = new LinkedHashMap<>(merges.size());
        for (int index = 0; index < merges.size(); index++) {
            ranks.put(merges.get(index), index);
        }
        return ranks;
    }
}
