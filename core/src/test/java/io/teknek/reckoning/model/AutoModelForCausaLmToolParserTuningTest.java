package io.teknek.reckoning.model;

import io.teknek.reckoning.generator.FinishReason;
import io.teknek.reckoning.generator.Response;
import io.teknek.reckoning.safetensors.fetch.ModelFetcher;
import io.teknek.reckoning.safetensors.prompt.ToolCall;
import io.teknek.reckoning.toolcallparser.ToolCallParser;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class AutoModelForCausaLmToolParserTuningTest {

    @Test
    void antaresUsesXmlToolCallParserTuning() {
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(new ModelFetcher("edwardcapriolo", "antares-1b-JQ4"));
        ToolCallParser parser = builder.toolCallParserForTest();

        List<ToolCall> calls = parser.extract(new Response("", """
                <think>inspect</think>
                <tool_call>
                {"name":"terminal","arguments":{"command":"rg -n Runtime"}}
                </tool_call>
                """, FinishReason.TOOL_CALLS, 0, List.of(), 0, 0, List.of()));

        assertEquals(1, calls.size());
        assertEquals("terminal", calls.getFirst().getName());
        assertEquals(Map.of("command", "rg -n Runtime"), calls.getFirst().getParameters());
    }
}
