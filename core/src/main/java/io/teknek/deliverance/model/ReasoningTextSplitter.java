package io.teknek.deliverance.model;

/** Splits model text into user-visible content and hidden reasoning delimited by {@code <think>} tags. */
public final class ReasoningTextSplitter {
    private static final String THINK_START = "<think>";
    private static final String THINK_END = "</think>";

    private final StringBuilder pending = new StringBuilder();
    private boolean inReasoning;

    public ReasoningTextSplitter() {
        this(false);
    }

    public ReasoningTextSplitter(boolean inReasoning) {
        this.inReasoning = inReasoning;
    }

    public static boolean promptEndsInsideReasoning(String prompt) {
        if (prompt == null) {
            return false;
        }
        return prompt.stripTrailing().endsWith(THINK_START);
    }

    public Part accept(String text) {
        pending.append(text == null ? "" : text);
        StringBuilder content = new StringBuilder();
        StringBuilder reasoning = new StringBuilder();
        while (pending.length() > 0) {
            if (inReasoning) {
                int end = indexOf(pending, THINK_END);
                if (end >= 0) {
                    reasoning.append(pending.substring(0, end));
                    pending.delete(0, end + THINK_END.length());
                    inReasoning = false;
                    continue;
                }
                int keep = longestPrefixSuffix(pending, THINK_END);
                int flush = pending.length() - keep;
                if (flush > 0) {
                    reasoning.append(pending.substring(0, flush));
                    pending.delete(0, flush);
                }
                break;
            }
            int start = indexOf(pending, THINK_START);
            if (start >= 0) {
                content.append(pending.substring(0, start));
                pending.delete(0, start + THINK_START.length());
                inReasoning = true;
                continue;
            }
            int keep = longestPrefixSuffix(pending, THINK_START);
            int flush = pending.length() - keep;
            if (flush > 0) {
                content.append(pending.substring(0, flush));
                pending.delete(0, flush);
            }
            break;
        }
        return new Part(content.toString(), reasoning.toString());
    }

    private static int longestPrefixSuffix(CharSequence text, String prefix) {
        int max = Math.min(text.length(), prefix.length() - 1);
        for (int length = max; length > 0; length--) {
            boolean matches = true;
            for (int i = 0; i < length; i++) {
                if (text.charAt(text.length() - length + i) != prefix.charAt(i)) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                return length;
            }
        }
        return 0;
    }

    private static int indexOf(CharSequence text, String needle) {
        for (int i = 0; i <= text.length() - needle.length(); i++) {
            boolean matches = true;
            for (int j = 0; j < needle.length(); j++) {
                if (text.charAt(i + j) != needle.charAt(j)) {
                    matches = false;
                    break;
                }
            }
            if (matches) {
                return i;
            }
        }
        return -1;
    }

    public record Part(String content, String reasoning) {
    }
}
