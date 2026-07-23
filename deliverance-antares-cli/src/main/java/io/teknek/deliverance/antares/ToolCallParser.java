package io.teknek.deliverance.antares;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class ToolCallParser {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final Pattern WRAPPED = Pattern.compile("<tool_call>\\s*(.*?)\\s*</tool_call>", Pattern.DOTALL | Pattern.CASE_INSENSITIVE);
    private static final Pattern RAW_TOOL_OBJECT = Pattern.compile("\\{\\s*\\\"(?:name|tool)\\\"\\s*:", Pattern.DOTALL);

    List<ToolCall> parse(String text) {
        List<ToolCall> calls = new ArrayList<>();
        Matcher wrapped = WRAPPED.matcher(text);
        while (wrapped.find()) {
            ToolCall call = parseObjectLenient(wrapped.group(1));
            if (call != null) {
                calls.add(call);
            }
        }
        if (!calls.isEmpty()) {
            return calls;
        }
        Matcher raw = RAW_TOOL_OBJECT.matcher(text);
        while (raw.find()) {
            String candidate = text.substring(raw.start());
            ToolCall call = parseObjectLenient(candidate);
            if (call != null) {
                calls.add(call);
            }
        }
        return calls;
    }

    static String cleanAssistantText(String text) {
        return WRAPPED.matcher(text)
                .replaceAll("")
                .replace("<|end_of_text|>", "")
                .replace("<|endoftext|>", "")
                .replace("<|eot_id|>", "")
                .replace("<think>", "")
                .replace("</think>", "")
                .trim();
    }

    private ToolCall parseObjectLenient(String raw) {
        String text = raw.trim();
        if (!text.startsWith("{")) {
            int start = text.indexOf('{');
            if (start < 0) {
                return null;
            }
            text = text.substring(start);
        }
        Integer balancedEnd = balancedObjectEnd(text);
        if (balancedEnd != null) {
            ToolCall call = parseExact(text.substring(0, balancedEnd));
            if (call != null) {
                return call;
            }
        }
        for (int end = text.length(); end > 0; end--) {
            int objectEnd = text.lastIndexOf('}', end - 1);
            if (objectEnd < 0) {
                return null;
            }
            ToolCall call = parseExact(text.substring(0, objectEnd + 1));
            if (call != null) {
                return call;
            }
            end = objectEnd;
        }
        return null;
    }

    private ToolCall parseExact(String candidate) {
        try {
            JsonNode node = JSON.readTree(candidate);
            return toToolCall(node);
        } catch (JsonProcessingException ignored) {
            return null;
        }
    }

    private Integer balancedObjectEnd(String text) {
        boolean inString = false;
        boolean escaped = false;
        int depth = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (escaped) {
                escaped = false;
                continue;
            }
            if (c == '\\' && inString) {
                escaped = true;
                continue;
            }
            if (c == '"') {
                inString = !inString;
                continue;
            }
            if (inString) {
                continue;
            }
            if (c == '{') {
                depth++;
            } else if (c == '}') {
                depth--;
                if (depth == 0) {
                    return i + 1;
                }
            }
        }
        return null;
    }

    private ToolCall toToolCall(JsonNode node) {
        if (node == null || !node.isObject()) {
            return null;
        }
        JsonNode nameNode = node.has("name") ? node.get("name") : node.get("tool");
        if (nameNode == null && node.has("ranked_files")) {
            return new ToolCall("submit_vulnerable_files", toMap(node));
        }
        if (nameNode == null || !nameNode.isTextual()) {
            return null;
        }
        JsonNode argsNode = node.has("arguments") ? node.get("arguments") : node.get("args");
        String name = nameNode.asText().trim().toLowerCase(Locale.ROOT).replaceAll("[\\s-]+", "_");
        if (argsNode == null && "submit_vulnerable_files".equals(name) && node.has("ranked_files")) {
            return new ToolCall(name, toMap(node));
        }
        if (argsNode == null && "submit_no_vulnerability_found".equals(name)) {
            return new ToolCall(name, Map.of());
        }
        if (argsNode == null || !argsNode.isObject()) {
            return null;
        }
        return new ToolCall(name, toMap(argsNode));
    }

    private Map<String, Object> toMap(JsonNode node) {
        Map<String, Object> result = new LinkedHashMap<>();
        Iterator<Map.Entry<String, JsonNode>> fields = node.fields();
        while (fields.hasNext()) {
            Map.Entry<String, JsonNode> field = fields.next();
            result.put(field.getKey(), toValue(field.getValue()));
        }
        return result;
    }

    private Object toValue(JsonNode node) {
        if (node.isTextual()) {
            return node.asText();
        }
        if (node.isInt() || node.isLong()) {
            return node.asLong();
        }
        if (node.isArray()) {
            List<Object> values = new ArrayList<>();
            node.forEach(value -> values.add(toValue(value)));
            return values;
        }
        if (node.isObject()) {
            return toMap(node);
        }
        if (node.isBoolean()) {
            return node.asBoolean();
        }
        return node.toString();
    }
}
