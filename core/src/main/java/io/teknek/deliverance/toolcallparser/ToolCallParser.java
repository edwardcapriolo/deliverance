package io.teknek.deliverance.toolcallparser;

import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.ResponseContext;
import io.teknek.deliverance.safetensors.prompt.ToolCall;

import java.util.List;
import java.util.Optional;

/**
 * This is a messy little interface. We have a text stream of token from a model,
 * that text stream has turns. The turns are signified differently in different models.
 * Also the end user with ChatCompletion is looking for a certain shape.
 * We will punt on having a model here, that understands the model returned via http
 * Instead just do a simple parsing and move on.
 *
 */
public interface ToolCallParser {
    /**
     *
     * @param response the response directly from the AbstractModel
     * @return a list of all tool calls found in the response
     */
    List<ToolCall> extract(Response response);


    /**
     * Some models (llama) use special tokens to signify the end of the response.
     * See <a href="https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/">...</a>
     * <|eot_id|> .. This token signals to the executor that the model has finished generating a response.
     * This means the interface needs to enter the control flow as the typical generation will not stop. It will
     * produce. And the request will end with max_tokens not tools call.
     * |eot_id|><|start_header_id|><|start_header_id|><|start_header_id|>....
     *
     * @return Some if the content of the response dictate the turn should end,
     */
    Optional<Response> shouldEndTurn(ResponseContext response, int length);
}
