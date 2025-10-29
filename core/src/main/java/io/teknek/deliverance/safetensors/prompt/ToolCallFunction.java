package io.teknek.deliverance.safetensors.prompt;

import java.util.LinkedHashMap;
import java.util.Map;

public class ToolCallFunction {
    private final ToolCall call;

    public ToolCallFunction(ToolCall call) {
        this.call = call;
    }

    public InnerToolCall function() {
        return new InnerToolCall(call);
    }

    public Map toMap() {
        Map<String, Object> args = new LinkedHashMap<>();
        args.put("name", call.getName());
        args.put("arguments", call.getParameters());
        return Map.of("function", args, "id", call.getId());
    }
}
