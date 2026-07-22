package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.FinishReason;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.ToolCall;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class AutoModelForCausaLmToolParserTuningTest {

    @Test
    void antaresUsesXmlToolCallParserTuning() {
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(new ModelFetcher("fdtn-ai", "antares-1b-JQ4"));
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
