package io.teknek.deliverance.tokenizer;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

public class PatternModel {
    public final java.util.regex.Pattern regex;

    @JsonCreator
    public PatternModel(@JsonProperty("Regex") String regex) {
        this.regex = java.util.regex.Pattern.compile(regex);
    }
}
