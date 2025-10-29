package io.teknek.deliverance.tokenizer;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Map;

public class NormalizerItem {
    public final String type;
    public final String prepend;
    public final Map<String, String> pattern;
    public final String content;

    @JsonCreator
    public NormalizerItem(
            @JsonProperty("type") String type,
            @JsonProperty("prepend") String prepend,
            @JsonProperty("pattern") Map<String, String> pattern,
            @JsonProperty("content") String content) {
        this.type = type;
        this.prepend = prepend;
        this.pattern = pattern;
        this.content = content;
    }

    public String normalize(String sentence) {
        switch (type) {
            case "Replace":
                return replace(sentence);
            case "Prepend":
                return prepend(sentence);
            case "NFC":
            case "NFKC":
            case "NFD":
            case "NFKD":
                return formNormalize(sentence);
            default:
                throw new IllegalArgumentException("Invalid normalizer type: " + type);
        }
    }

    private String formNormalize(String sentence) {
        //thread safe reuse?
        java.text.Normalizer.Form form = java.text.Normalizer.Form.valueOf(type);
        return java.text.Normalizer.normalize(sentence, form);
    }

    private String replace(String sentence) {
        for (Map.Entry<String, String> entry : pattern.entrySet()) {
            if (!entry.getKey().equalsIgnoreCase("String")) {
                System.out.println("Ignoring unknown pattern key: " + entry.getKey());
            }
            sentence = sentence.replaceAll(entry.getValue(), content);
        }
        return sentence;
    }

    private String prepend(String sentence) {
        return prepend + sentence;
    }
}