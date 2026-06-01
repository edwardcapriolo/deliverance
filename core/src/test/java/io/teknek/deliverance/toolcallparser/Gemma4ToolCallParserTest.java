package io.teknek.deliverance.toolcallparser;

import io.teknek.deliverance.generator.FinishReason;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.ResponseContext;
import io.teknek.deliverance.model.SamplerReturn;
import io.teknek.deliverance.safetensors.prompt.ToolCall;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class Gemma4ToolCallParserTest {

    @Test
    void extractGemma4ToolCall() {
        String responseText = "<|channel>thought\nNeed weather data<channel|><|tool_call>call:get_weather{\"location\":\"Albany, NY\",\"unit\":\"celsius\"}<tool_call|>";
        Response response = new Response("", responseText, FinishReason.TOOL_CALLS, 0, null,
                0, 0, List.of(new SamplerReturn(0)));

        Gemma4ToolCallParser parser = new Gemma4ToolCallParser();
        List<ToolCall> toolCalls = parser.extract(response);

        assertEquals(1, toolCalls.size());
        ToolCall toolCall = toolCalls.get(0);
        assertEquals("101", toolCall.getId());
        assertEquals("get_weather", toolCall.getName());
        assertEquals("Albany, NY", toolCall.getParameters().get("location"));
        assertEquals("celsius", toolCall.getParameters().get("unit"));
    }

    @Test
    void extractGemma4ToolCallWithDottedName() {
        String responseText = "<|tool_call>call:flight_searcher.get_weather{\"location\":\"Albany, NY\"}<tool_call|>";
        Response response = new Response("", responseText, FinishReason.TOOL_CALLS, 0, null,
                0, 0, List.of(new SamplerReturn(0)));

        Gemma4ToolCallParser parser = new Gemma4ToolCallParser();
        List<ToolCall> toolCalls = parser.extract(response);

        assertEquals(1, toolCalls.size());
        ToolCall toolCall = toolCalls.get(0);
        assertEquals("flight_searcher.get_weather", toolCall.getName());
        assertEquals("Albany, NY", toolCall.getParameters().get("location"));
    }

    @Test
    void shouldEndTurnOnceGemma4ToolCallCloses() {
        Gemma4ToolCallParser parser = new Gemma4ToolCallParser();

        ResponseContext openContext = Mockito.mock(ResponseContext.class);
        Mockito.when(openContext.getResponseTextWithSpecialTokens()).thenReturn(
                new StringBuilder("<|tool_call>call:get_weather{\"location\":\"Albany, NY\"}"));
        assertTrue(parser.shouldEndTurn(openContext, 0).isEmpty());

        ResponseContext closedContext = Mockito.mock(ResponseContext.class);
        Mockito.when(closedContext.getResponseText()).thenReturn(new StringBuilder(""));
        Mockito.when(closedContext.getResponseTextWithSpecialTokens()).thenReturn(
                new StringBuilder("<|tool_call>call:get_weather{\"location\":\"Albany, NY\"}<tool_call|>"));
        Mockito.when(closedContext.getGeneratedTokens()).thenReturn(Collections.singletonList(1));

        Response response = parser.shouldEndTurn(closedContext, 0).orElseThrow();
        assertEquals(FinishReason.TOOL_CALLS, response.finishReason);
        assertEquals(1, response.toolCalls.size());
        assertEquals("get_weather", response.toolCalls.get(0).getName());
        assertEquals("Albany, NY", response.toolCalls.get(0).getParameters().get("location"));
    }
}
