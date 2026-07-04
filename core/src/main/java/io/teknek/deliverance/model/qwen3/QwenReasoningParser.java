package io.teknek.deliverance.model.qwen3;

public final class QwenReasoningParser {
    private static final String THINK_START = "<think>";
    private static final String THINK_END = "</think>";

    private QwenReasoningParser() {
    }

    public static Parsed parse(String responseWithSpecialTokens, String fallbackContent) {
        if (responseWithSpecialTokens == null || responseWithSpecialTokens.isEmpty()) {
            return new Parsed(cleanContent(fallbackContent), null);
        }
        int start = responseWithSpecialTokens.indexOf(THINK_START);
        if (start < 0) {
            return new Parsed(cleanContent(fallbackContent), null);
        }
        int end = responseWithSpecialTokens.indexOf(THINK_END, start + THINK_START.length());
        if (end < 0) {
            return new Parsed(cleanContent(fallbackContent), null);
        }
        String reasoning = responseWithSpecialTokens.substring(start + THINK_START.length(), end).trim();
        String withoutReasoning = responseWithSpecialTokens.substring(0, start)
                + responseWithSpecialTokens.substring(end + THINK_END.length());
        String content = cleanContent(withoutReasoning);
        if (content.isEmpty()) {
            content = cleanContent(fallbackContent);
        }
        return new Parsed(content, reasoning.isEmpty() ? null : reasoning);
    }

    private static String cleanContent(String input) {
        if (input == null || input.isEmpty()) {
            return "";
        }
        return input.replace(THINK_START, "").replace(THINK_END, "").trim();
    }

    public record Parsed(String content, String reasoning) {
    }
}
