package io.teknek.deliverance.antares;

final class StreamingConsoleRenderer {
    private static final String GREY = "\u001B[90m";
    private static final String CYAN = "\u001B[36m";
    private static final String RESET = "\u001B[0m";

    private final boolean colorEnabled;
    private final StringBuilder buffer = new StringBuilder();
    private boolean thinking;
    private boolean suppressingToolCall;
    private boolean lineStarted;
    private String activeColor = "";

    StreamingConsoleRenderer() {
        this(System.getenv("NO_COLOR") == null && !"dumb".equalsIgnoreCase(System.getenv("TERM")));
    }

    StreamingConsoleRenderer(boolean colorEnabled) {
        this.colorEnabled = colorEnabled;
    }

    void assistantChunk(String chunk) {
        buffer.append(chunk);
        drain(false);
    }

    void endAssistantTurn() {
        drain(true);
        resetColor();
        if (lineStarted) {
            System.err.println();
            lineStarted = false;
        }
    }

    void toolCall(ToolCall call) {
        resetColor();
        String details = switch (call.name()) {
            case "terminal", "bash" -> String.valueOf(call.arguments().getOrDefault("command", ""));
            case "read_file" -> String.valueOf(call.arguments().getOrDefault("path", ""));
            default -> call.arguments().toString();
        };
        printLabel("[tool] " + call.name() + (details.isBlank() ? "" : " " + details));
    }

    void toolResult(String response) {
        resetColor();
        String text = response == null ? "" : response.stripTrailing();
        if (text.isBlank()) {
            printLabel("[result] <empty output>");
            return;
        }
        if (text.length() > 2_000) {
            text = text.substring(0, 2_000) + "\n[result truncated in console; full output was sent to the model]";
        }
        if (lineStarted) {
            System.err.println();
            lineStarted = false;
        }
        setColor(GREY);
        System.err.println("[result]\n" + text);
        resetColor();
    }

    void assistantNotice(String message) {
        resetColor();
        printLabel("[assistant] " + message);
    }

    private void drain(boolean flush) {
        while (buffer.length() > 0) {
            if (suppressingToolCall) {
                int close = indexOfIgnoreCase(buffer, "</tool_call>");
                if (close < 0) {
                    if (flush) {
                        buffer.setLength(0);
                    }
                    return;
                }
                buffer.delete(0, close + "</tool_call>".length());
                suppressingToolCall = false;
                continue;
            }

            int tagStart = buffer.indexOf("<");
            if (tagStart < 0) {
                if (flush) {
                    printText(buffer.toString());
                    buffer.setLength(0);
                } else if (buffer.length() > 32) {
                    printText(buffer.substring(0, buffer.length() - 32));
                    buffer.delete(0, buffer.length() - 32);
                }
                return;
            }
            if (tagStart > 0) {
                printText(buffer.substring(0, tagStart));
                buffer.delete(0, tagStart);
                continue;
            }
            String tag = nextKnownTag();
            if (tag == null) {
                if (!flush && buffer.length() < 64) {
                    return;
                }
                printText(buffer.substring(0, 1));
                buffer.delete(0, 1);
                continue;
            }
            consumeTag(tag);
        }
    }

    private String nextKnownTag() {
        String lower = buffer.toString().toLowerCase();
        for (String tag : new String[]{"<think>", "</think>", "<tool_call>", "</tool_call>",
                "<|end_of_text|>", "<|endoftext|>", "<|eot_id|>"}) {
            if (lower.startsWith(tag)) {
                return tag;
            }
        }
        return null;
    }

    private void consumeTag(String tag) {
        buffer.delete(0, tag.length());
        switch (tag) {
            case "<think>" -> thinking = true;
            case "</think>" -> thinking = false;
            case "<tool_call>" -> suppressingToolCall = true;
            default -> {
            }
        }
    }

    private void printText(String text) {
        if (text.isEmpty()) {
            return;
        }
        setColor(thinking ? GREY : "");
        System.err.print(text);
        System.err.flush();
        lineStarted = true;
    }

    private void printLabel(String text) {
        if (lineStarted) {
            System.err.println();
            lineStarted = false;
        }
        setColor(CYAN);
        System.err.print(text);
        resetColor();
        System.err.println();
    }

    private void setColor(String color) {
        if (!colorEnabled) {
            return;
        }
        if (!activeColor.equals(color)) {
            if (!activeColor.isEmpty()) {
                System.err.print(RESET);
            }
            if (!color.isEmpty()) {
                System.err.print(color);
            }
            activeColor = color;
        }
    }

    private void resetColor() {
        if (colorEnabled && !activeColor.isEmpty()) {
            System.err.print(RESET);
            activeColor = "";
        }
    }

    private int indexOfIgnoreCase(StringBuilder sb, String needle) {
        return sb.toString().toLowerCase().indexOf(needle.toLowerCase());
    }
}
