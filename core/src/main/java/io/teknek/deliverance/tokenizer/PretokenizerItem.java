package io.teknek.deliverance.tokenizer;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.google.common.annotations.VisibleForTesting;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;

public class PretokenizerItem {

    public static final String DIGITS = "Digits";
    public static final String SPLIT = "Split";
    public static final String BYTE_LEVEL = "ByteLevel";

    public final String type;
    public final PatternModel pattern;
    public final String behavior;
    public final Boolean invert;
    public final Boolean individual_digits;
    public final Boolean add_prefix_space;
    public final Boolean trim_offsets;
    public final Boolean use_regex;

    @JsonCreator
    public PretokenizerItem(
            @JsonProperty("type") String type,
            @JsonProperty("pattern") PatternModel pattern,
            @JsonProperty("behavior") String behavior,
            @JsonProperty("invert") Boolean invert,
            @JsonProperty("individual_digits") Boolean individual_digits,
            @JsonProperty("add_prefix_space") Boolean add_prefix_space,
            @JsonProperty("trim_offsets") Boolean trim_offsets,
            @JsonProperty("use_regex") Boolean use_regex
    ) {
        this.type = type;
        this.pattern = pattern;
        this.behavior = behavior;
        this.invert = invert;
        this.individual_digits = individual_digits;
        this.add_prefix_space = add_prefix_space;
        this.trim_offsets = trim_offsets;
        this.use_regex = use_regex;
    }

    public List<String> pretokenize(String sentence) {
        return switch (type) {
            case "Split" -> splitRegex(sentence);
            case "Digits" -> splitDigits(sentence);
            case "ByteLevel" ->
                    throw new IllegalArgumentException("suspicious renable");
            default -> throw new IllegalArgumentException("Invalid pre-tokenizer type: " + type);
        };
    }

    @VisibleForTesting
    List<String> splitRegex(String s) {
        Matcher m = pattern.regex.matcher(s);
        List<String> ret = new ArrayList<>();
        int start = 0;
        while (m.find()) {
            String r = s.substring(start, m.start());
            if (!r.isEmpty()) ret.add(r);

            ret.add(m.group());
            start = m.end();
        }

        String p = start >= s.length() ? "" : s.substring(start);
        if (!p.isEmpty()) ret.add(p);
        return ret;
    }

    @VisibleForTesting
    static List<String> splitDigits(String sentence) {
        return List.of(sentence.split("(?<=\\D)(?=\\d)|(?<=\\d)(?=\\D)"));
    }
}
