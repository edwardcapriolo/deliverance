package io.teknek.deliverance.tokenizer;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class PreTokenizer {
    public final String type;
    public final String replacement;
    public final String prependScheme;
    public final boolean isLegacy;
    public final List<PretokenizerItem> pretokenizers;

    @JsonCreator
    public PreTokenizer(
            @JsonProperty("type") String type,
            @JsonProperty("replacement") String replacement,
            @JsonProperty("prepend_scheme") String prependScheme,
            @JsonProperty("pretokenizers") List<PretokenizerItem> pretokenizers
    ) {
        this.type = type;
        this.replacement = replacement;
        this.prependScheme = prependScheme;
        this.pretokenizers = pretokenizers == null ? Collections.emptyList() : ImmutableList.copyOf(pretokenizers);
        this.isLegacy = this.pretokenizers.stream().map(p -> p.type).anyMatch(t -> t.equals("ByteLevel"));
    }

    public List<String> pretokenize(String sentence) {
        if (type.equalsIgnoreCase("MetaSpace")) {
            if (prependScheme.equalsIgnoreCase("first")) {
                sentence = " " + sentence;
            }
            return Collections.singletonList(sentence.replaceAll("[ \t]+", replacement));
        }
        if (pretokenizers.isEmpty()) {
            return Collections.singletonList(sentence);
        }

        Preconditions.checkArgument(type.equalsIgnoreCase("Sequence"), "Invalid pre-tokenizer type: " + type);
        List<String> pieces = List.of(sentence);
        List<String> tmp = new ArrayList<>();
        for (PretokenizerItem item : pretokenizers) {
            for (String piece : pieces) {
                tmp.addAll(item.pretokenize(piece));
            }

            pieces = tmp;
            tmp = new ArrayList<>();
        }

        return pieces;
    }
}
