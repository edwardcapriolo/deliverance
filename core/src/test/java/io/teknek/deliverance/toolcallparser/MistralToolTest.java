package io.teknek.deliverance.toolcallparser;

import io.teknek.deliverance.generator.FinishReason;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.SamplerReturn;
import io.teknek.deliverance.safetensors.prompt.ToolCall;
import org.junit.jupiter.api.Disabled;

import java.util.HashMap;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MistralToolTest {

    String s ="[AVAILABLE_TOOLS] [{\"type\": \"function\", \"function\": {\"name\": \"get_current_temperature\", \"description\": \"Simulates getting the current temperature at a location.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"The location to get the temperature for, in the format \\\"City, Country\\\".\"}, \"unit\": {\"type\": \"string\", \"description\": \"The unit to return the temperature in (e.g., \\\"celsius\\\", \\\"fahrenheit\\\").\"}}, \"required\": [\"location\", \"unit\"]}}}, {\"type\": \"function\", \"function\": {\"name\": \"get_current_temperature\", \"description\": \"Simulates getting the current temperature at a location.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"The location to get the temperature for, in the format \\\"City, Country\\\".\"}, \"unit\": {\"type\": \"string\", \"description\": \"The unit to return the temperature in (e.g., \\\"celsius\\\", \\\"fahrenheit\\\").\"}}, \"required\": [\"location\", \"unit\"]}}}][/AVAILABLE_TOOLS][INST] What is the temperature";

    @Disabled
    void flipCoinResponse(){
        Response r = new Response("", s, FinishReason.TOOL_CALLS,0, null,
                0, 0, List.of(new SamplerReturn(1)));
        LlamaToolCallParser c = new LlamaToolCallParser();
        List<ToolCall> resp = c.extract(r);
        assertEquals(1, resp.size());
        ToolCall expectedTool = resp.get(0);
        assertEquals( "101", expectedTool.getId());
        assertEquals( "get_current_temperature", expectedTool.getName());
        assertEquals( new HashMap<String,Object>(), expectedTool.getParameters());
    }
}
