package io.teknek.deliverance.toolcallparser;

import io.teknek.deliverance.generator.FinishReason;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.SamplerReturn;
import io.teknek.deliverance.safetensors.prompt.ToolCall;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class QuenToolParserTest {


    String multipleTools = """
            <tools>{"name": "get_match_schedule", "arguments": {"location": "San Jose, California, USA"}}{"name": "get_current_temperature", "arguments": {"location": "San Jose, California, USA"}}</tools>
            """;

    @Test
    public void testParseQWen() throws IOException, InterruptedException {
        Response r = new Response("", multipleTools, FinishReason.TOOL_CALLS,0, null,
                0, 0, List.of(new SamplerReturn(0)));
        QwenToolCallParser  c = new QwenToolCallParser ();
        List<ToolCall> resp = c.extract(r);
        assertEquals(2, resp.size());
        {
            ToolCall expectedTool = resp.get(0);
            assertEquals("101", expectedTool.getId());
            assertEquals("get_match_schedule", expectedTool.getName());
            assertEquals("San Jose, California, USA", expectedTool.getParameters().get("location"));
        }

        {
            ToolCall expectedTool = resp.get(1);
            assertEquals("102", expectedTool.getId());
            assertEquals("get_current_temperature", expectedTool.getName());
            assertEquals("San Jose, California, USA", expectedTool.getParameters().get("location"));
        }
    }

    @Test
    public void testParseQwenToolCallTag() {
        String toolCall = """
                <tool_call>
                {"name": "java_sandbox", "arguments": {"files": {"main.java": "class HelloWorld {}"}, "mainClass": "HelloWorld"}}
                </tool_call>
                """;
        Response r = new Response("", toolCall, FinishReason.TOOL_CALLS, 0, null,
                0, 0, List.of(new SamplerReturn(0)));
        QwenToolCallParser c = new QwenToolCallParser();

        List<ToolCall> resp = c.extract(r);

        assertEquals(1, resp.size());
        ToolCall expectedTool = resp.get(0);
        assertEquals("101", expectedTool.getId());
        assertEquals("java_sandbox", expectedTool.getName());
        assertEquals("HelloWorld", expectedTool.getParameters().get("mainClass"));
    }

    @Test
    public void jsonStringsTest(){
        QwenToolCallParser  c = new QwenToolCallParser ();
        String s = """
                {"A":"B", { } } { }
                """;
        List<String> actual = c.jsonStrings(s);
        assertEquals(2, actual.size());
        assertEquals("{\"A\":\"B\", { } }", actual.get(0));
        assertEquals(" { }", actual.get(1));
    }
}
