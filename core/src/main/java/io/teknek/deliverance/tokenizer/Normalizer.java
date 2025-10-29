package io.teknek.deliverance.tokenizer;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.collect.ImmutableList;

import java.util.Collections;
import java.util.List;

public class Normalizer {
    private final String type;

    private final List<NormalizerItem> normalizerItems;

    @JsonCreator
    public Normalizer(@JsonProperty("type") String type, @JsonProperty("normalizers") List<NormalizerItem> normalizerItems) {
        this.type = type;
        this.normalizerItems = normalizerItems == null ? Collections.emptyList() : ImmutableList.copyOf(normalizerItems);
    }

    public List<NormalizerItem> getNormalizerItems() {
        return normalizerItems;
    }

    public String getType() {
        return type;
    }

    public String normalize(String sentence) {
        if (normalizerItems.isEmpty()) {
            return sentence;
        }
        for (NormalizerItem item : normalizerItems) {
            sentence = item.normalize(sentence);
        }
        return sentence;
    }
}
