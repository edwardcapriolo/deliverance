package io.teknek.deliverance.safetensors.prompt;

import com.fasterxml.jackson.annotation.JsonPropertyOrder;

import java.util.Map;

@JsonPropertyOrder({ ToolResult.JSON_PROPERTY_TOOL_NAME, ToolResult.JSON_PROPERTY_TOOL_ID, ToolResult.JSON_PROPERTY_RESULT })
public class ToolResult {
    public static final String JSON_PROPERTY_TOOL_NAME = "name";
    public final String name;

    public static final String JSON_PROPERTY_TOOL_ID = "tool_call_id";
    public final String id;

    public static final String JSON_PROPERTY_RESULT = "result";
    private final Object result;

    private ToolResult(String name, String id, Object result) {
        this.name = name;
        this.id = id;
        this.result = result;
    }

    public static ToolResult from(String name, String id, Object result) {
        return new ToolResult(name, id, result);
    }

    public Object getResult() {
        return result;
    }

    public String getName() {
        return name;
    }

    public String getToolCallId() {
        return id;
    }

    public Map<String, Object> toJson() {
        return Map.of("content", Map.of("result", result));
    }
}
