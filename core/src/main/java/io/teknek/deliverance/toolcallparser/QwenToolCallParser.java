package io.teknek.deliverance.toolcallparser;

import com.fasterxml.jackson.core.JsonProcessingException;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.ResponseContext;
import io.teknek.deliverance.safetensors.prompt.ToolCall;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class QwenToolCallParser implements ToolCallParser {

    private static final Logger LOG = LoggerFactory.getLogger(LlamaToolCallParser.class);

    //"name": "get_match_schedule", "arguments": {"location": "San Jose, California, USA"}}
    public static class QwenFunction {
        public QwenFunction(){}
        public String name;
        public Map<String, Object> arguments;

        @Override
        public String toString() {
            return "QuenFunction{" +
                    "name='" + name + '\'' +
                    ", arguments=" + arguments +
                    '}';
        }
    }

    @Override
    public List<ToolCall> extract(Response response) {
        List<ToolCall> result = new ArrayList<>();
        extractTagged(response.responseTextWithSpecialTokens, "<tools>", "</tools>", result);
        extractTagged(response.responseTextWithSpecialTokens, "<tool_call>", "</tool_call>", result);
        AtomicInteger id = new AtomicInteger(101);
        List<ToolCall> distinct = result.stream().distinct().toList();
        distinct.forEach(x -> x.setId((id.getAndIncrement()) + ""));
        return distinct;
    }

    private void extractTagged(String text, String startTag, String endTag, List<ToolCall> result) {
        int index = text.indexOf(startTag);
        do {
            if (index == -1) {
                return;
            }
            int end = text.indexOf(endTag, index);
            if (end == -1) {
                break;
            }
            String section = text.substring(index + startTag.length(), end);
            List<String> jsons = jsonStrings(section);

            for (String json:jsons){
                try {
                    QwenFunction function = JsonUtils.om.readValue(json, QwenFunction.class);
                    if (function.name != null ){
                        ToolCall toolCall = new ToolCall(function.name, function.arguments);
                        result.add(toolCall);
                    }
                } catch (JsonProcessingException e) {
                    LOG.warn("Attempting to parse function: ", e);
                }
            }
            index = text.indexOf(startTag, end + endTag.length());
        } while (index < text.length());
    }

    @Override
    public Optional<Response> shouldEndTurn(ResponseContext response, int length) {
        return Optional.empty();
    }

    public List<String> jsonStrings(String input){
        Deque<Character> d = new ArrayDeque<>();
        StringBuilder jsonString = new StringBuilder();
        List<String> allJson = new ArrayList<>();
        //note this doesnt cover \} ifyou care send a pr
        for (Character c: input.toCharArray()){
            if (c == '{'){
                jsonString.append(c);
                d.push(c);
            } else if (c == '}'){
                jsonString.append(c);
                if (d.isEmpty()){
                    return Collections.emptyList();
                }
                char ignore = d.pop();
                if (d.isEmpty()){
                    allJson.add(jsonString.toString());
                    jsonString = new StringBuilder();
                }
            } else {
                jsonString.append(c);
            }
        }
        return allJson;
    }
}
