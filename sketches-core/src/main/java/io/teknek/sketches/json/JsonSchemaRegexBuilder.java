package io.teknek.sketches.json;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Converts JSON Schema documents into regular expressions for guided generation.
 *
 * <p>This class is intentionally being ported in small steps from outlines-core. The primitive regex fragments are present
 * first so tests and later schema-node compilation can build on stable names. Full schema-to-regex conversion is not
 * implemented yet.</p>
 */
public final class JsonSchemaRegexBuilder {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    public static final String WHITESPACE = "[ \\n\\t\\r]*";
    public static final String NULL = "null";
    public static final String BOOLEAN = "(true|false)";
    public static final String INTEGER = "(-)?(0|[1-9][0-9]*)";
    public static final String NUMBER = "((-)?((0|[1-9][0-9]*)(\\.[0-9]+)?|\\.[0-9]+)([eE][+-]?[0-9]+)?)";
    public static final String STRING_INNER = "([^\"\\\\]|\\\\([\"\\\\/bfnrt]|u[0-9a-fA-F]{4}))*";
    public static final String STRING = "\"" + STRING_INNER + "\"";
    public static final String DATE = "[0-9]{4}-[0-9]{2}-[0-9]{2}";
    public static final String TIME = "[0-9]{2}:[0-9]{2}:[0-9]{2}(\\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})?";
    public static final String DATE_TIME = DATE + "T" + TIME;
    public static final String EMAIL = "[^@\\s]+@[^@\\s]+\\.[^@\\s]+";
    public static final String URI = "[a-zA-Z][a-zA-Z0-9+.-]*:[^\\s]*";
    public static final String UUID = "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}";

    private JsonSchemaRegexBuilder() {
    }

    public static String buildRegexFromSchema(String schemaJson) {
        return buildRegexFromSchema(schemaJson, WHITESPACE);
    }

    public static String buildRegexFromSchema(String schemaJson, String whitespacePattern) {
        Objects.requireNonNull(schemaJson, "schemaJson");
        Objects.requireNonNull(whitespacePattern, "whitespacePattern");
        JsonNode schema;
        try {
            schema = OBJECT_MAPPER.readTree(schemaJson);
        } catch (Exception e) {
            throw new IllegalArgumentException("Expected a valid JSON string.", e);
        }
        return regexForSchema(schema, whitespacePattern);
    }

    private static String regexForSchema(JsonNode schema, String whitespacePattern) {
        if (schema.has("const")) {
            return regexForConst(schema.get("const"));
        }
        if (schema.has("enum")) {
            return regexForEnum(schema.get("enum"));
        }
        if (schema.has("anyOf")) {
            return regexForSchemaAlternatives(schema.get("anyOf"), whitespacePattern, "anyOf");
        }
        if (schema.has("oneOf")) {
            return regexForSchemaAlternatives(schema.get("oneOf"), whitespacePattern, "oneOf");
        }
        String type = schema.path("type").asText();
        return switch (type) {
            case "object" -> regexForObject(schema, whitespacePattern);
            case "string" -> STRING;
            case "integer" -> INTEGER;
            case "number" -> NUMBER;
            case "boolean" -> BOOLEAN;
            case "null" -> NULL;
            case "array" -> regexForArray(schema, whitespacePattern);
            default -> throw new UnsupportedOperationException("Unsupported JSON Schema type: " + type);
        };
    }

    private static String regexForObject(JsonNode schema, String whitespacePattern) {
        JsonNode properties = schema.path("properties");
        if (!properties.isObject()) {
            return "\\{" + whitespacePattern + "\\}";
        }

        List<String> propertyRegexes = new ArrayList<>();
        Iterator<String> fieldNames = properties.fieldNames();
        while (fieldNames.hasNext()) {
            String fieldName = fieldNames.next();
            JsonNode propertySchema = properties.get(fieldName);
            propertyRegexes.add(jsonStringLiteral(fieldName)
                    + whitespacePattern
                    + ":"
                    + whitespacePattern
                    + regexForSchema(propertySchema, whitespacePattern));
        }

        return "\\{" + whitespacePattern + String.join(whitespacePattern + "," + whitespacePattern, propertyRegexes)
                + whitespacePattern + "\\}";
    }

    private static String regexForArray(JsonNode schema, String whitespacePattern) {
        JsonNode itemSchema = schema.path("items");
        if (itemSchema.isMissingNode()) {
            throw new UnsupportedOperationException("Array schemas without items are not supported yet");
        }
        String itemRegex = regexForSchema(itemSchema, whitespacePattern);
        return "\\[" + whitespacePattern + "(" + itemRegex + "(" + whitespacePattern + "," + whitespacePattern
                + itemRegex + ")*)?" + whitespacePattern + "\\]";
    }

    private static String regexForSchemaAlternatives(JsonNode alternativesNode, String whitespacePattern, String keyword) {
        if (!alternativesNode.isArray()) {
            throw new IllegalArgumentException("JSON Schema " + keyword + " must be an array");
        }
        List<String> alternatives = new ArrayList<>();
        for (JsonNode alternative : alternativesNode) {
            alternatives.add(regexForSchema(alternative, whitespacePattern));
        }
        return "(" + String.join("|", alternatives) + ")";
    }

    private static String regexForEnum(JsonNode enumNode) {
        if (!enumNode.isArray()) {
            throw new IllegalArgumentException("JSON Schema enum must be an array");
        }
        List<String> choices = new ArrayList<>();
        for (JsonNode value : enumNode) {
            choices.add(regexForConst(value));
        }
        return "(" + String.join("|", choices) + ")";
    }

    private static String regexForConst(JsonNode value) {
        if (value.isTextual()) {
            return jsonStringLiteral(value.asText());
        }
        if (value.isIntegralNumber()) {
            return escapeLiteral(value.asText());
        }
        if (value.isFloatingPointNumber()) {
            return escapeLiteral(value.asText());
        }
        if (value.isBoolean()) {
            return value.asBoolean() ? "true" : "false";
        }
        if (value.isNull()) {
            return NULL;
        }
        throw new UnsupportedOperationException("Unsupported const value: " + value);
    }

    private static String jsonStringLiteral(String value) {
        try {
            String json = OBJECT_MAPPER.writeValueAsString(value);
            return escapeLiteral(json);
        } catch (Exception e) {
            throw new IllegalArgumentException("Could not render JSON string literal", e);
        }
    }

    private static String escapeLiteral(String value) {
        StringBuilder escaped = new StringBuilder(value.length() * 2);
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            if ("\\|.?*+(){}[]^$\"".indexOf(c) >= 0) {
                escaped.append('\\');
            }
            escaped.append(c);
        }
        return escaped.toString();
    }
}
