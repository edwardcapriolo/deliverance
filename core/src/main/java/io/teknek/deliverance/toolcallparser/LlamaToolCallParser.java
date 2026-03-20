package io.teknek.deliverance.toolcallparser;
import com.fasterxml.jackson.core.JsonProcessingException;

import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.safetensors.prompt.ToolCall;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.util.Optional;
import java.util.regex.Pattern;


public class LlamaToolCallParser implements ToolCallParser {

    private static final Logger LOG = LoggerFactory.getLogger(LlamaToolCallParser.class);
    String eot = "<|eot_id|>";

    public MessageAndToolCall extract(Response response){
        int id = 101;

        MessageAndToolCall cr = new MessageAndToolCall();
        String [] parts = response.responseTextWithSpecialTokens.split(Pattern.quote(eot));
        for(String part : parts){
            if (part.startsWith("assistant<|end_header_id|>")){
                Optional<String> x = extractJsonSubstring(part);
                if (x.isPresent()) {
                    try {
                        ToolCall toolCall = JsonUtils.om.readValue(x.get(), ToolCall.class);
                        toolCall.setId((id++) + "");
                        cr.messages.add(new MessageDatum("assistant", null, toolCall));
                    } catch (JsonProcessingException e) {
                        LOG.warn("Attempting to parse tool call:", e );
                        cr.messages.add(new MessageDatum("assistant", part, null));
                    }
                }
            } else {
                cr.messages.add(new MessageDatum("assistant", part, null));
            }
        }

        return cr;
    }

    public static Optional<String> extractJsonSubstring(String mixedContent) {
        int startIndex = mixedContent.indexOf('{');
        int endIndex = mixedContent.lastIndexOf('}');
        if (startIndex != -1 && endIndex != -1 && endIndex > startIndex) {
            return Optional.of(mixedContent.substring(startIndex, endIndex + 1));
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
