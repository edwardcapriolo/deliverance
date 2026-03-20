package io.teknek.deliverance.toolcallparser;

import io.teknek.deliverance.safetensors.prompt.ToolCall;

public class MessageDatum {
    public String type;
    public String content;
    public ToolCall toolCall;

    public MessageDatum(String type, String message, ToolCall toolCall) {
        this.type = type;
        this.content = message;
        this.toolCall = toolCall;
    }
}
