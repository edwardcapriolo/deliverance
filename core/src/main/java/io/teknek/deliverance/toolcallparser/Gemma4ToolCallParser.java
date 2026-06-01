package io.teknek.deliverance.toolcallparser;

import com.fasterxml.jackson.core.JsonProcessingException;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.generator.FinishReason;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.ResponseContext;
import io.teknek.deliverance.safetensors.prompt.ToolCall;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Parses Gemma 4 tool calls of the form {@code <|tool_call>call:name{...}<tool_call|>}.
 *
 * <p>This follows the tool-call shape described in the Gemma tokenizer metadata, where the tool-call payload is
 * wrapped in tool-call sentinels and the inner text contains {@code call:} followed by the function name and a
 * JSON object of arguments.</p>
 */
public class Gemma4ToolCallParser implements ToolCallParser {

    private static final Logger LOG = LoggerFactory.getLogger(Gemma4ToolCallParser.class);
    static final String TOOL_CALL_START = "<|tool_call>";
    static final String TOOL_CALL_END = "<tool_call|>";
    private static final Pattern TOOL_CALL_SECTION = Pattern.compile(
            Pattern.quote(TOOL_CALL_START) + "(.*?)" + Pattern.quote(TOOL_CALL_END),
            Pattern.DOTALL
    );
    private static final Pattern TOOL_CALL_PAYLOAD = Pattern.compile(
            "call:(?<name>[^\\{\\s]+)\\s*(?<arguments>\\{.*})",
            Pattern.DOTALL
    );

    @Override
    public List<ToolCall> extract(Response response) {
        Matcher sectionMatcher = TOOL_CALL_SECTION.matcher(response.responseTextWithSpecialTokens);
        List<ToolCall> result = new ArrayList<>();
        while (sectionMatcher.find()) {
            String payload = sectionMatcher.group(1).trim();
            Matcher payloadMatcher = TOOL_CALL_PAYLOAD.matcher(payload);
            if (!payloadMatcher.matches()) {
                continue;
            }
            String name = payloadMatcher.group("name");
            String argumentsJson = payloadMatcher.group("arguments");
            try {
                @SuppressWarnings("unchecked")
                var arguments = JsonUtils.om.readValue(argumentsJson, java.util.Map.class);
                result.add(new ToolCall(name, arguments));
            } catch (JsonProcessingException e) {
                LOG.warn("Attempting to parse Gemma 4 tool call payload: {}", payload, e);
            }
        }
        AtomicInteger id = new AtomicInteger(101);
        List<ToolCall> distinct = result.stream().distinct().toList();
        distinct.forEach(x -> x.setId(String.valueOf(id.getAndIncrement())));
        return distinct;
    }

    @Override
    public Optional<Response> shouldEndTurn(ResponseContext response, int length) {
        if (response.getResponseTextWithSpecialTokens().indexOf(TOOL_CALL_END) == -1) {
            return Optional.empty();
        }
        Response parsed = new Response(
                response.getResponseText().toString(),
                response.getResponseTextWithSpecialTokens().toString(),
                FinishReason.TOOL_CALLS,
                length,
                response.getGeneratedTokens(),
                0,
                0,
                response.samplerReturnList
        );
        return Optional.of(parsed.copyWithToolCalls(extract(parsed)));
    }
}
