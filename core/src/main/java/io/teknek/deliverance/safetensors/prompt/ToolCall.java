package io.teknek.deliverance.safetensors.prompt;

import com.fasterxml.jackson.annotation.*;

import java.util.Map;
import java.util.Objects;

@JsonPropertyOrder({ ToolResult.JSON_PROPERTY_TOOL_NAME, ToolResult.JSON_PROPERTY_TOOL_ID })
public class ToolCall {
    @JsonProperty("name")
    private final String name;

    @JsonProperty("id")
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private String id;

    @JsonProperty("parameters")
    private final Map<String, Object> parameters;

    @JsonCreator
    public ToolCall(
            @JsonProperty("name") String name,
            @JsonAlias({ "arguments" }) @JsonProperty("parameters") Map<String, Object> parameters
    ) {
        this.name = name;
        this.parameters = parameters;
    }

    public ToolCall(String name, String id, Map<String, Object> parameters) {
        this.name = name;
        this.id = id;
        this.parameters = parameters;
    }

    public String getName() {
        return name;
    }

    public Map<String, Object> getParameters() {
        return parameters;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getId() {
        return id;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ToolCall toolCall = (ToolCall) o;
        return Objects.equals(name, toolCall.name) && Objects.equals(parameters, toolCall.parameters);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, parameters);
    }
}
