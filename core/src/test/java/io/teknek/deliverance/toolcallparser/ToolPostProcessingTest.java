package io.teknek.deliverance.toolcallparser;

import io.teknek.deliverance.generator.FinishReason;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.safetensors.prompt.ToolCall;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ToolPostProcessingTest {

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
}
