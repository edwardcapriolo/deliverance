package io.teknek.reckoning.safetensors.prompt.llmon;

/** Wire syntax used by the Granite LLMON prompt format. */
public final class LlmOnSyntax {
    public static final String OPEN = "<|open|>";
    public static final String CLOSE = "<|close|>";
    public static final String OPEN_END = "<|open_end|>";
    public static final String SELF_CLOSE = "<|self_close|>";
    public static final String DOT = "<|.|>";
    public static final String COLON = "<|:|>";
    public static final String TOOL_CALL_OPEN = "<tool_call>";
    public static final String TOOL_CALL_CLOSE = "</tool_call>";
    public static final String TOOL_RESPONSE_OPEN = "<tool_response>";
    public static final String TOOL_RESPONSE_CLOSE = "</tool_response>";
    public static final String EOS = "<|end_of_text|>";
    /** Granite's 75 single-token characters used to form two-character block IDs. */
    public static final String ID_CHARS = "¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¹º»¼½¾¿ÀÁÂÃÄÇÉÍÎÐÑÓÖ×ÚÜßàáâãäåæçèéêëìíîïðñòóôõöøùúûüý";

    private LlmOnSyntax() {
    }

    public static String reference(String kind, String id) {
        return kind + COLON + id;
    }
}
