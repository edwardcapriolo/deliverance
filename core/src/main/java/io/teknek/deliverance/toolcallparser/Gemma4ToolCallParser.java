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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
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
                var arguments = parseArguments(argumentsJson);
                result.add(new ToolCall(name, arguments));
            } catch (RuntimeException e) {
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

    @SuppressWarnings("unchecked")
    private static Map<String, Object> parseArguments(String value) {
        try {
            return JsonUtils.om.readValue(value, Map.class);
        } catch (JsonProcessingException ignored) {
            Object parsed = new GemmaArgumentParser(value).parseValue();
            if (!(parsed instanceof Map<?, ?> map)) {
                throw new IllegalArgumentException("Gemma4 arguments must parse to an object");
            }
            return (Map<String, Object>) map;
        }
    }

    private static final class GemmaArgumentParser {
        private static final String STRING_DELIMITER = "<|\"|>";
        private final String input;
        private int pos;

        private GemmaArgumentParser(String input) {
            this.input = input;
        }

        private Object parseValue() {
            skipWhitespace();
            if (peek('{')) {
                return parseObject();
            }
            if (peek('[')) {
                return parseArray();
            }
            if (input.startsWith(STRING_DELIMITER, pos)) {
                return parseDelimitedString();
            }
            String atom = parseAtom();
            return switch (atom) {
                case "true" -> Boolean.TRUE;
                case "false" -> Boolean.FALSE;
                case "null" -> null;
                default -> parseNumberOrString(atom);
            };
        }

        private Map<String, Object> parseObject() {
            expect('{');
            Map<String, Object> map = new LinkedHashMap<>();
            skipWhitespace();
            if (peek('}')) {
                pos++;
                return map;
            }
            while (true) {
                String key = parseKey();
                expect(':');
                map.put(key, parseValue());
                skipWhitespace();
                if (peek('}')) {
                    pos++;
                    return map;
                }
                expect(',');
            }
        }

        private List<Object> parseArray() {
            expect('[');
            List<Object> list = new ArrayList<>();
            skipWhitespace();
            if (peek(']')) {
                pos++;
                return list;
            }
            while (true) {
                list.add(parseValue());
                skipWhitespace();
                if (peek(']')) {
                    pos++;
                    return list;
                }
                expect(',');
            }
        }

        private String parseKey() {
            skipWhitespace();
            return parseAtom();
        }

        private String parseDelimitedString() {
            pos += STRING_DELIMITER.length();
            int end = input.indexOf(STRING_DELIMITER, pos);
            if (end < 0) {
                throw new IllegalArgumentException("Unterminated Gemma4 string");
            }
            String value = input.substring(pos, end);
            pos = end + STRING_DELIMITER.length();
            return value;
        }

        private String parseAtom() {
            skipWhitespace();
            int start = pos;
            while (pos < input.length()) {
                char c = input.charAt(pos);
                if (c == ',' || c == '}' || c == ']' || c == ':' || Character.isWhitespace(c)) {
                    break;
                }
                pos++;
            }
            if (start == pos) {
                throw new IllegalArgumentException("Expected atom at " + pos);
            }
            return input.substring(start, pos);
        }

        private Object parseNumberOrString(String atom) {
            try {
                return Integer.parseInt(atom);
            } catch (NumberFormatException ignored) {
                try {
                    return Double.parseDouble(atom);
                } catch (NumberFormatException ignoredAgain) {
                    return atom;
                }
            }
        }

        private boolean peek(char expected) {
            skipWhitespace();
            return pos < input.length() && input.charAt(pos) == expected;
        }

        private void expect(char expected) {
            skipWhitespace();
            if (pos >= input.length() || input.charAt(pos) != expected) {
                throw new IllegalArgumentException("Expected '" + expected + "' at " + pos);
            }
            pos++;
        }

        private void skipWhitespace() {
            while (pos < input.length() && Character.isWhitespace(input.charAt(pos))) {
                pos++;
            }
        }
    }
}
