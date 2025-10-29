package io.teknek.deliverance.safetensors.prompt;

import java.util.Map;

public class InnerToolCall {
    private final ToolCall call;

    public InnerToolCall(ToolCall call) {
        this.call = call;
    }

    public Map<String, Object> arguments() {
        return call.getParameters();
    }

    public String name() {
        return call.getName();
    }
}
