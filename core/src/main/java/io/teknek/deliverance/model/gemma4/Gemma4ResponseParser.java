package io.teknek.deliverance.model.gemma4;

public final class Gemma4ResponseParser {
    private static final String THOUGHT_START = "<|channel>thought\n";
    private static final String CHANNEL_END = "<channel|>";

    private Gemma4ResponseParser() {
    }

    public static Parsed parse(String responseWithSpecialTokens, String fallbackContent) {
        if (responseWithSpecialTokens == null || responseWithSpecialTokens.isEmpty()) {
            return new Parsed(cleanContent(fallbackContent), null);
        }

        int thoughtStart = responseWithSpecialTokens.indexOf(THOUGHT_START);
        if (thoughtStart < 0) {
            return new Parsed(cleanContent(fallbackContent), null);
        }

        int thoughtEnd = responseWithSpecialTokens.indexOf(CHANNEL_END, thoughtStart + THOUGHT_START.length());
        if (thoughtEnd < 0) {
            return new Parsed(cleanContent(fallbackContent), null);
        }

        String reasoning = responseWithSpecialTokens.substring(thoughtStart + THOUGHT_START.length(), thoughtEnd).trim();
        String withoutThoughts = responseWithSpecialTokens.substring(0, thoughtStart)
                + responseWithSpecialTokens.substring(thoughtEnd + CHANNEL_END.length());
        String content = cleanContent(withoutThoughts);
        if (content.isEmpty()) {
            content = cleanContent(fallbackContent);
        }
        return new Parsed(content, reasoning.isEmpty() ? null : reasoning);
    }

    private static String cleanContent(String input) {
        if (input == null || input.isEmpty()) {
            return "";
        }
        return input
                .replace("<|channel>", "")
                .replace("<channel|>", "")
                .replace("<|turn>", "")
                .replace("<turn|>", "")
                .replace("<|think|>", "")
                .trim();
    }

    public record Parsed(String content, String reasoning) {
    }
}
