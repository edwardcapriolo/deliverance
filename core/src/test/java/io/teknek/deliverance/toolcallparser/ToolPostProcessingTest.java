package io.teknek.deliverance.toolcallparser;

import io.teknek.deliverance.generator.FinishReason;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.prompt.ToolCall;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.mockito.Mockito;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ToolPostProcessingTest {

    /*
     Qwen 2.5 7B return:

    {'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'role': 'assistant', 'content': '<tools>\n{"name": "get_match_schedule", "arguments": {"location": "San Jose, California, USA"}}\n{"name": "get_current_temperature", "arguments": {"location": "San Jose, California, USA"}}\n</tools>'}}]

Other models:

    {'choices': [{'finish_reason': 'tool_calls', 'index': 0, 'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'type': 'function', 'function': {'name': 'get_match_schedule', 'arguments': '{"location":"San Jose, California, USA"}'}, 'id': ''}]}}]

     */


    String flipCoinNoParamLlamaResponse = """
             {"name": "flip_coin", "parameters": {}}<|eot_id|>assistant<|end_header_id|> {"name": "flip_coin", "parameters": {}}<|eot_id|>
            
            The output should be: {"name": "flip_coin", "parameters": {}}<|eot_id|>
            
            This is because the prompt is asking to decide who goes first by a coin flip, which means we need to flip a coin to determine the outcome. 
            The function `flip_coin` is used to simulate a coin flip, and since we want to decide who goes first, 
            we need to call this function to get the result. The output of the function will be either "H" 
            for heads or "T" for tails, which will determine who goes first. |<|end_header_id|>
            """;

    @Test
    void flipCoinResponse(){
        Response r = new Response("", flipCoinNoParamLlamaResponse, FinishReason.TOOL_CALL,0, null,
                0, 0);
        LlamaToolCallParser c = new LlamaToolCallParser();
        List<ToolCall> resp = c.extract(r);
        assertEquals(1, resp.size());
        ToolCall expectedTool = resp.get(0);
        assertEquals( "101", expectedTool.getId());
        assertEquals( "flip_coin", expectedTool.getName());
        assertEquals( new HashMap<String,Object>(), expectedTool.getParameters());
    }


    @Test
    void temperatureTest(){
        String s = """
    {"type":"function","function":"get_current_temperature","parameters":{"location":"New York, USA","unit":"celsius"}}<|end_header_id|>This will return the current temperature in New York City in Celsius. If you want the temperature in Fahrenheit, you can change the unit to "fahrenheit".<|end_header_id|>Note: I've assumed that the function `get_current_temperature` is already defined and available in the environment. If it's not, you'll need to define it or use a different function to get the current temperature.""";
        Response r = new Response("", s, FinishReason.TOOL_CALL,0, null,
                0, 0);
        LlamaToolCallParser c = new LlamaToolCallParser();
        List<ToolCall> resp = c.extract(r);
        assertEquals(1, resp.size());
        ToolCall expectedTool = resp.get(0);
        assertEquals( "101", expectedTool.getId());
        assertEquals( "get_current_temperature", expectedTool.getName());
        assertEquals( "New York, USA", expectedTool.getParameters().get("location"));
        assertEquals( "celsius", expectedTool.getParameters().get("unit"));
    }

    @Test
    void shouldEndFlowTest(){
        String missingEot = """
                {"type":"function","function":"get_current_temperature","parameters":{"location":"New York, USA","unit":"celsius"}}<|end_header_id|>This will return the current temperature in New York City in Celsius. If you want the temperature in Fahrenheit, you can change the unit to "fahrenheit".<|end_header_id|>Note: I've assumed that the function `get_current_temperature` is already defined and available in the environment. If it's not, you would need to define it or import it from a library.', responseTextWithSpecialTokens='{"type":"function","function":"get_current_temperature","parameters":{"location":"New York, USA","unit":"celsius"}}<|end_header_id|>This will return the current temperature in New York City in Celsius. If you want the temperature in Fahrenheit, you can change the unit to "fahrenheit".<|end_header_id|>Note: I've assumed that the function `get_current_temperature` is already defined and available in the environment. If it's not, you would need to define it or import it from a library.
                """;
        {
            LlamaToolCallParser c = new LlamaToolCallParser();
            AbstractModel.ResponseContext context = Mockito.mock(AbstractModel.ResponseContext.class);
            Mockito.when(context.getResponseTextWithSpecialTokens()).thenReturn(new StringBuilder(missingEot));
            Assertions.assertEquals(Optional.empty(), c.shouldEndTurn(context, 0));
        }
        {

            String withEot = """
                    {"type":"function","function":"get_current_temperature","parameters":{"location":"New York, USA","unit":"celsius"}}<|end_header_id|>This will return the current temperature in New York City in Celsius. If you want the temperature in Fahrenheit, you can change the unit to "fahrenheit".<|end_header_id|>Note: I've assumed that the function `get_current_temperature` is already defined and available in the environment. If it's not, you would need to define it or import it from a library.', responseTextWithSpecialTokens='{"type":"function","function":"get_current_temperature","parameters":{"location":"New York, USA","unit":"celsius"}}<|end_header_id|>This will return the current temperature in New York City in Celsius. If you want the temperature in Fahrenheit, you can change the unit to "fahrenheit".<|end_header_id|>Note: I've assumed that the function `get_current_temperature` is already defined and available in the environment. If it's not, you would need to define it or import it from a library.<|eot_id|>""";
            LlamaToolCallParser c = new LlamaToolCallParser();
            AbstractModel.ResponseContext context = Mockito.mock(AbstractModel.ResponseContext.class);
            Mockito.when(context.getResponseText()).thenReturn(new StringBuilder("bla"));
            Mockito.when(context.getResponseTextWithSpecialTokens()).thenReturn(new StringBuilder(withEot));
            Mockito.when(context.getGeneratedTokens()).thenReturn(Collections.singletonList(1));
            Response x = c.shouldEndTurn(context, 0).get();
            Assertions.assertEquals(FinishReason.TOOL_CALL, x.finishReason);
            Assertions.assertEquals(1, x.toolCalls.size());
        }
    }
}
