package io.teknek.deliverance.toolcallparser;

import com.fasterxml.jackson.core.JsonProcessingException;

import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.generator.*;

import io.teknek.deliverance.model.ResponseContext;
import io.teknek.deliverance.safetensors.prompt.ToolCall;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Pattern;

class Function {
    public String type;
    public String function;
    public Map<String, Object> parameters;
}

public class LlamaToolCallParser implements ToolCallParser {

    private static final Logger LOG = LoggerFactory.getLogger(LlamaToolCallParser.class);
    String eot = "<|eot_id|>";
    String head = "<|end_header_id|>";

    public List<ToolCall> extract(Response response) {
        AtomicInteger id = new AtomicInteger(101);
        String[] parts = response.responseTextWithSpecialTokens.split(Pattern.quote(head));
        List<ToolCall> result = new ArrayList<>();
        for (String part : parts) {
            Optional<String> x = extractJsonSubstring(part);
            if (x.isPresent()) {
                try {
                    Function function = JsonUtils.om.readValue(x.get(), Function.class);
                    if (function.type != null && function.function != null) {
                        ToolCall tc = new ToolCall(function.function, function.parameters);
                        result.add(tc);
                    }
                } catch (JsonProcessingException e) {
                    LOG.warn("Attempting to parse function: ", e);
                }
                try {
                    ToolCall toolCall = JsonUtils.om.readValue(x.get(), ToolCall.class);
                    if (toolCall.getName() != null) {
                        result.add(toolCall);
                    }
                } catch (JsonProcessingException e) {
                    LOG.warn("Attempting to parse tool call:", e);
                }
            }
        }
        List<ToolCall> distinct = result.stream().distinct().toList();
        distinct.forEach(x -> x.setId((id.getAndIncrement()) + ""));
        return distinct;
    }

    @Override
    public Optional<Response> shouldEndTurn(ResponseContext response, int length) {
        if(response.getResponseTextWithSpecialTokens().indexOf(eot) > -1){
            Response resp = new Response(response.getResponseText().toString(), response.getResponseTextWithSpecialTokens().toString(),
                    FinishReason.TOOL_CALLS,
                    length, response.getGeneratedTokens(), 0, 0, response.samplerReturnList);
            return Optional.of(resp.copyWithToolCalls(extract(resp)));
        }
        return Optional.empty();
    }

    public static Optional<String> extractJsonSubstring(String mixedContent) {
        int startIndex = mixedContent.indexOf('{');
        int endIndex = mixedContent.lastIndexOf('}');
        if (startIndex != -1 && endIndex != -1 && endIndex > startIndex) {
            String matched = mixedContent.substring(startIndex, endIndex + 1);
            return Optional.of(matched);
        } else {
            return Optional.empty();
        }
    }


}
/*
/*
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\"location\": \"Boston, MA\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "model": "gpt-4o",
  ...
 */
