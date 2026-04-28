package io.teknek.deliverance.grace;

public record PreTokenizerConfig(String splitPattern, boolean addPrefixSpace, boolean useRegex) {
    public static final String GPT2_REGEX = "(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

    public PreTokenizerConfig {
        splitPattern = splitPattern == null || splitPattern.isBlank() ? null : splitPattern;
    }

    public String effectiveSplitPattern() {
        if (splitPattern != null) {
            return splitPattern;
        }
        return useRegex ? GPT2_REGEX : null;
    }
}
