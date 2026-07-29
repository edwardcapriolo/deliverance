package io.teknek.sketches.json;

import dk.brics.automaton.RegExp;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JsonSchemaRegexBuilderTest {

    @Test
    void primitiveRegexConstantsArePresent() {
        List<String> constants = List.of(
                JsonSchemaRegexBuilder.BOOLEAN,
                JsonSchemaRegexBuilder.DATE,
                JsonSchemaRegexBuilder.DATE_TIME,
                JsonSchemaRegexBuilder.EMAIL,
                JsonSchemaRegexBuilder.INTEGER,
                JsonSchemaRegexBuilder.NULL,
                JsonSchemaRegexBuilder.NUMBER,
                JsonSchemaRegexBuilder.STRING,
                JsonSchemaRegexBuilder.STRING_INNER,
                JsonSchemaRegexBuilder.TIME,
                JsonSchemaRegexBuilder.URI,
                JsonSchemaRegexBuilder.UUID,
                JsonSchemaRegexBuilder.WHITESPACE
        );

        for (String constant : constants) {
            Pattern.compile(constant);
        }
        assertTrue(JsonSchemaRegexBuilder.WHITESPACE.isEmpty());
    }

    @Test
    void primitiveRegexConstantsMatchRepresentativeValues() {
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.NULL, "null"));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.BOOLEAN, "true"));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.BOOLEAN, "false"));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.INTEGER, "-42"));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.NUMBER, "3.14e10"));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.STRING, "\"hello\\nworld\""));
        assertFalse(Pattern.matches(JsonSchemaRegexBuilder.STRING, "\"hello\nworld\""));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.DATE, "2026-07-12"));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.DATE_TIME, "2026-07-12T10:30:00Z"));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.EMAIL, "a@example.com"));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.URI, "https://example.com"));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.UUID, "123e4567-e89b-12d3-a456-426614174000"));
    }

    @Test
    void jsonStringRejectsRawControlCharactersAndAcceptsEscapes() {
        assertFalse(Pattern.matches(JsonSchemaRegexBuilder.STRING, "\"bad\nvalue\""));
        assertFalse(Pattern.matches(JsonSchemaRegexBuilder.STRING, "\"bad\tvalue\""));
        assertFalse(Pattern.matches(JsonSchemaRegexBuilder.STRING, "\"bad\rvalue\""));
        assertFalse(Pattern.matches(JsonSchemaRegexBuilder.STRING, "\"bad" + ((char) 1) + "value\""));

        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.STRING, "\"bad\\nvalue\""));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.STRING, "\"bad\\tvalue\""));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.STRING, "\"bad\\rvalue\""));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.STRING, "\"say \\\"hi\\\"\""));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.STRING, "\"path \\\\tmp\""));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.STRING, "\"snowman \\u2603\""));

        assertFalse(Pattern.matches(JsonSchemaRegexBuilder.STRING, "\"bad\\xvalue\""));
        assertFalse(Pattern.matches(JsonSchemaRegexBuilder.STRING, "\"bad\\u12xz\""));
    }

    @Test
    void buildRegexFromJsonSchema() {
        String schema = """
                {
                  "type": "object",
                  "properties": {
                    "foo": { "type": "integer" },
                    "bar": { "type": "string" }
                  },
                  "required": ["foo", "bar"]
                }
                """;

        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(schema);

        assertTrue(Pattern.matches(regex, "{\"foo\":4,\"bar\":\"baz    baz baz bar\"}"));
        assertFalse(Pattern.matches(regex, "{\"foo\" : 4 ,\"bar\":\"baz    baz baz bar\"}"));
        assertCompilesWithBrics(regex);
    }

    @Test
    void defaultWhitespaceRequiresTightStructuralJson() {
        String schema = """
                {
                  "type": "object",
                  "properties": {
                    "places": { "type": "array", "items": { "type": "string" } }
                  }
                }
                """;

        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(schema);

        assertTrue(Pattern.matches(regex, "{\"places\":[]}"));
        assertFalse(Pattern.matches(regex, "{\"places\" n:[]}"));
        assertFalse(Pattern.matches(regex, "{\"places\" :[]}"));
        assertFalse(Pattern.matches(regex, "{\"places\": []}"));
        assertCompilesWithBrics(regex);
    }

    @Test
    void customWhitespaceCanAllowPrettyJsonWhenExplicitlyRequested() {
        String schema = """
                {
                  "type": "object",
                  "properties": {
                    "places": { "type": "array", "items": { "type": "string" } }
                  }
                }
                """;

        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(schema, "[ \\n\\t\\r]{0,16}");

        assertTrue(Pattern.matches(regex, "{\"places\"                :[]}"));
        assertFalse(Pattern.matches(regex, "{\"places\"                 :[]}"));
        assertFalse(Pattern.matches(regex, "{\"places\":                 []}"));
        assertCompilesWithBrics(regex);
    }

    @Test
    void buildRegexFromJsonSchemaWithCustomWhitespace() {
        String schema = """
                {
                  "type": "object",
                  "properties": {
                    "foo": { "type": "integer" },
                    "bar": { "type": "string" }
                  },
                  "required": ["foo", "bar"]
                }
                """;

        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(schema, "[\\n ]*");

        assertTrue(Pattern.matches(regex, "{     \"foo\"   :   4, \n\n\n   \"bar\": \"baz    baz baz bar\"\n\n}"));
        assertCompilesWithBrics(regex);
    }

    @Test
    void invalidJsonSchemaStringThrowsClearError() {
        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> JsonSchemaRegexBuilder.buildRegexFromSchema("{'name':"));

        assertTrue(error.getMessage().contains("Expected a valid JSON string."));
    }

    @Test
    void supportsNestedObjects() {
        String schema = """
                {
                  "type": "object",
                  "properties": {
                    "outer": {
                      "type": "object",
                      "properties": {
                        "inner": { "type": "integer" }
                      },
                      "required": ["inner"]
                    }
                  },
                  "required": ["outer"]
                }
                """;

        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(schema);

        assertTrue(Pattern.matches(regex, "{\"outer\":{\"inner\":7}}"));
        assertCompilesWithBrics(regex);
    }

    @Test
    void supportsArrays() {
        String schema = """
                {
                  "type": "array",
                  "items": { "type": "integer" }
                }
                """;

        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(schema);

        assertTrue(Pattern.matches(regex, "[]"));
        assertTrue(Pattern.matches(regex, "[1,2,-3]"));
        assertCompilesWithBrics(regex);
    }

    @Test
    void supportsArrayMinAndMaxItems() {
        String schema = """
                {
                  "type": "array",
                  "minItems": 2,
                  "maxItems": 3,
                  "items": { "type": "integer" }
                }
                """;

        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(schema);

        assertFalse(Pattern.matches(regex, "[]"));
        assertFalse(Pattern.matches(regex, "[1]"));
        assertTrue(Pattern.matches(regex, "[1,2]"));
        assertTrue(Pattern.matches(regex, "[1,2,3]"));
        assertFalse(Pattern.matches(regex, "[1,2,3,4]"));
        assertCompilesWithBrics(regex);
    }

    @Test
    void supportsExactArrayItemCount() {
        String schema = """
                {
                  "type": "array",
                  "minItems": 2,
                  "maxItems": 2,
                  "items": { "type": "string", "maxLength": 4 }
                }
                """;

        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(schema);

        assertFalse(Pattern.matches(regex, "[]"));
        assertFalse(Pattern.matches(regex, "[\"a\"]"));
        assertTrue(Pattern.matches(regex, "[\"a\",\"bc\"]"));
        assertFalse(Pattern.matches(regex, "[\"a\",\"bc\",\"d\"]"));
        assertFalse(Pattern.matches(regex, "[\"aaaaa\",\"bc\"]"));
        assertCompilesWithBrics(regex);
    }

    @Test
    void supportsStringMaxLength() {
        String schema = """
                { "type": "string", "maxLength": 5 }
                """;

        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(schema);

        assertTrue(Pattern.matches(regex, "\"\""));
        assertTrue(Pattern.matches(regex, "\"abcde\""));
        assertFalse(Pattern.matches(regex, "\"abcdef\""));
        assertCompilesWithBrics(regex);
    }

    @Test
    void supportsStringPattern() {
        String schema = """
                { "type": "string", "pattern": "[A-Za-z][A-Za-z ]{0,39}" }
                """;

        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(schema);

        assertTrue(Pattern.matches(regex, "\"cheese\""));
        assertTrue(Pattern.matches(regex, "\"ice cream\""));
        assertFalse(Pattern.matches(regex, "\"} food\""));
        assertFalse(Pattern.matches(regex, "\",ledger\""));
        assertFalse(Pattern.matches(regex, "\"西里\""));
        assertCompilesWithBrics(regex);
    }

    @Test
    void stringMaxLengthBoundsEscapedAtoms() {
        String schema = """
                { "type": "string", "maxLength": 2 }
                """;

        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(schema);

        assertTrue(Pattern.matches(regex, "\"ab\""));
        assertTrue(Pattern.matches(regex, "\"\\na\""));
        assertTrue(Pattern.matches(regex, "\"\\u2603a\""));
        assertFalse(Pattern.matches(regex, "\"abc\""));
        assertFalse(Pattern.matches(regex, "\"\\nab\""));
        assertCompilesWithBrics(regex);
    }

    @Test
    void objectStringMaxLengthBoundsNestedFields() {
        String schema = """
                {
                  "type": "object",
                  "properties": {
                    "whyCluesMatter": { "type": "string", "maxLength": 12 }
                  }
                }
                """;

        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(schema);

        assertTrue(Pattern.matches(regex, "{\"whyCluesMatter\":\"short\"}"));
        assertFalse(Pattern.matches(regex, "{\"whyCluesMatter\":\"methodClueMattersMistakeClue\"}"));
        assertCompilesWithBrics(regex);
    }

    @Test
    void supportsEnumAndConst() {
        String enumSchema = """
                { "enum": ["red", "blue", 7, true, null] }
                """;
        String constSchema = """
                { "const": "fixed" }
                """;

        String enumRegex = JsonSchemaRegexBuilder.buildRegexFromSchema(enumSchema);
        String constRegex = JsonSchemaRegexBuilder.buildRegexFromSchema(constSchema);

        assertTrue(Pattern.matches(enumRegex, "\"red\""));
        assertTrue(Pattern.matches(enumRegex, "7"));
        assertTrue(Pattern.matches(enumRegex, "true"));
        assertTrue(Pattern.matches(enumRegex, "null"));
        assertTrue(Pattern.matches(constRegex, "\"fixed\""));
        assertCompilesWithBrics(enumRegex);
        assertCompilesWithBrics(constRegex);
    }

    @Test
    void supportsAnyOfAndOneOf() {
        String schema = """
                {
                  "anyOf": [
                    { "type": "integer" },
                    { "const": "unknown" }
                  ]
                }
                """;
        String oneOfSchema = """
                {
                  "oneOf": [
                    { "type": "boolean" },
                    { "type": "null" }
                  ]
                }
                """;

        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.buildRegexFromSchema(schema), "42"));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.buildRegexFromSchema(schema), "\"unknown\""));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.buildRegexFromSchema(oneOfSchema), "false"));
        assertTrue(Pattern.matches(JsonSchemaRegexBuilder.buildRegexFromSchema(oneOfSchema), "null"));
        assertCompilesWithBrics(JsonSchemaRegexBuilder.buildRegexFromSchema(schema));
        assertCompilesWithBrics(JsonSchemaRegexBuilder.buildRegexFromSchema(oneOfSchema));
    }

    @Test
    void deadToRightsCaseSchemaCompilesWithBricsAutomaton() {
        String schema = """
                {
                  "type": "object",
                  "additionalProperties": false,
                  "required": ["caseTitle", "suspect", "setting", "clues", "hiddenTruth"],
                  "properties": {
                    "caseTitle": { "type": "string" },
                    "suspect": { "type": "string" },
                    "setting": { "type": "string" },
                    "clues": { "type": "array", "minItems": 3, "maxItems": 3, "items": { "type": "string" } },
                    "hiddenTruth": {
                      "type": "object",
                      "additionalProperties": false,
                      "required": ["crime", "motive", "method", "mistakes", "whyCluesMatter", "confession"],
                      "properties": {
                        "crime": { "type": "string" },
                        "motive": { "type": "string" },
                        "method": { "type": "string" },
                        "mistakes": { "type": "array", "items": { "type": "string" } },
                        "whyCluesMatter": { "type": "array", "items": { "type": "string" } },
                        "confession": { "type": "string" }
                      }
                    }
                  }
                }
                """;

        String regex = JsonSchemaRegexBuilder.buildRegexFromSchema(schema);

        assertCompilesWithBrics(regex);
    }

    private static void assertCompilesWithBrics(String regex) {
        new RegExp(regex).toAutomaton();
    }
}
