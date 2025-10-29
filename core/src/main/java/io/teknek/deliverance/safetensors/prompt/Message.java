package io.teknek.deliverance.safetensors.prompt;

import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class Message {
    private final Object content;
    private final PromptRole role;
    private final ToolCallFunction toolCalls;
    private final String toolCallId;

    public Message(Object content, PromptRole role) {
        this.content = content;
        this.role = role;
        this.toolCalls = null;
        this.toolCallId = null;
    }

    public Message(ToolCall toolCall) {
        this.content = null;
        this.role = PromptRole.TOOL_CALL;
        this.toolCalls = new ToolCallFunction(toolCall);
        this.toolCallId = toolCall.getId();
    }

    public Message(ToolResult toolResult) {
        this.content = toolResult.toJson();
        this.toolCalls = null;
        this.role = PromptRole.TOOL;
        this.toolCallId = toolResult.getToolCallId();
    }

    public Object getContent() {
        return content;
    }

    public Map toMap() {
        Map map = new HashMap();
        map.put("role", role.name().toLowerCase());
        map.put("content", content == null ? "" : content);
        if (toolCalls != null) {
            map.put("tool_calls", List.of(toolCalls.toMap()));
        }
        if (toolCallId != null) {
            map.put("tool_call_id", toolCallId);
        }
        return map;
    }

    public String getRole() {
        return role.name().toLowerCase();
    }


    public List<ToolCallFunction> toolCalls() {
        if (toolCalls == null) {
            return null;
        }
        return List.of(toolCalls);
    }
}