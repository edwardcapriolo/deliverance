package io.teknek.deliverance.antares;

record ToolResult(boolean submitted, String response, AgentResult result) {
    static ToolResult response(String response) {
        return new ToolResult(false, response, null);
    }

    static ToolResult submitted(AgentResult result) {
        return new ToolResult(true, result.summary(), result);
    }
}
