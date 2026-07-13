# Guided JSON Plan

## Goal

Add guided JSON generation by compiling JSON Schema into the existing sketches regex/index/guide pipeline.

The intended path is:

```text
guided_json request
  -> JsonSchema term
  -> JSON Schema to regex
  -> Index(regex, Vocabulary)
  -> IndexGuide
  -> GuideLogitsProcessor
  -> sampler masks invalid logits
```

## Existing Foundation

Already available:

- `sketches-core` module.
- `Term` types including `JsonSchema`.
- `Vocabulary`.
- Automaton-backed `Index`.
- `IndexGuide`.
- `GuideLogitsProcessor` in core.
- `GeneratorParameters.withGuidedRegex(...)`.
- Web/OpenAPI `guided_regex`.
- Server-side `SketchesSettings` limits:
  - `maxRegexLength`
  - `maxIndexStates`
  - `maxIndexTransitions`

Guided JSON should reuse this path rather than adding a separate sampler or JSON-specific logits path.

## Public API Shape

Add a vLLM-style chat-completion request field:

```json
{
  "guided_json": {
    "type": "object",
    "properties": {
      "name": { "type": "string" },
      "age": { "type": "integer" }
    },
    "required": ["name", "age"],
    "additionalProperties": false
  }
}
```

Core API:

```java
GeneratorParameters.withGuidedJson(String jsonSchema)
```

The web layer can accept `guided_json` as an object and serialize it to the internal schema string, or accept a string if the generated OpenAPI model makes object passthrough awkward. Prefer object support in the OpenAPI schema.

Guidance modes are mutually exclusive:

```text
guidedChoice
guidedRegex
guidedJson
```

Only one can be set for a request.

## Test-First Port

Start by porting upstream tests from:

```text
outlines-core/tests/test_json_schema.py
```

Initial Java test file:

```text
sketches-core/src/test/java/io/teknek/sketches/json/JsonSchemaRegexBuilderTest.java
```

Ported tests:

- `test_build_regex_from_json_schema`
  - schema with object fields:
    - `foo: integer`
    - `bar: string`
  - `buildRegexFromSchema(schema)` matches compact-ish JSON.
  - `buildRegexFromSchema(schema, "[\n ]*")` matches JSON with whitespace.
- `test_invalid_json`
  - invalid schema JSON throws a clear exception.
- `test_types_presence_and_not_emptyness`
  - primitive regex fragments are non-empty:
    - `BOOLEAN`
    - `INTEGER`
    - `NUMBER`
    - `NULL`
    - `STRING`
    - `STRING_INNER`
    - `DATE`
    - `DATE_TIME`
    - `EMAIL`
    - `TIME`
    - `URI`
    - `UUID`
    - `WHITESPACE`

Use these tests to drive the implementation. Avoid custom-only tests until upstream behavior is covered.

## JSON Schema To Regex Builder

Add:

```text
sketches-core/src/main/java/io/teknek/sketches/json/JsonSchemaRegexBuilder.java
```

Initial API:

```java
public final class JsonSchemaRegexBuilder {
    public static String buildRegexFromSchema(String schemaJson);
    public static String buildRegexFromSchema(String schemaJson, String whitespacePattern);
}
```

Initial supported schema features:

- `type: "object"`
- `properties`
- `required`
- `additionalProperties: false`
- `type: "string"`
- `type: "integer"`
- `type: "number"`
- `type: "boolean"`
- `type: "null"`
- `enum`
- `const`
- arrays if covered by ported upstream tests later

Start with the minimal subset needed for upstream `test_json_schema.py`, then expand by porting more tests.

## Object Property Ordering

JSON objects are semantically unordered, but regex is sequential.

For generation, use deterministic property order from the schema. This emits a valid JSON text satisfying the schema, not every possible textual ordering a validator would accept.

Avoid property permutation expansion initially because it can explode combinatorially.

## Core Wiring

After `JsonSchemaRegexBuilder` tests pass:

1. Add `GeneratorParameters.guidedJson`.
2. Add `GeneratorParameters.withGuidedJson(String jsonSchema)`.
3. Extend `LogitsProcessorFactory`:

```text
guidedJson
  -> JsonSchemaRegexBuilder.buildRegexFromSchema(schema)
  -> Vocabulary from model
  -> Index(regex, vocabulary, sketchesSettings)
  -> IndexGuide
  -> GuideLogitsProcessor
```

4. Add conflict validation against `guidedChoice` and `guidedRegex`.

## Web/OpenAPI Wiring

Add `guided_json` to `CreateChatCompletionRequest` in:

```text
web/deliverance_specification.yaml
```

Then regenerate sources:

```sh
mvn -pl web -DskipTests generate-sources
```

Map the request in `ChatCompletionService`:

```text
request.getGuidedJson() -> params.withGuidedJson(schemaJson)
```

Add web mapping tests in `ChatCompletionControllerTest`.

## Processor Tests

Add focused tests similar to guided regex:

```text
core/src/test/java/io/teknek/deliverance/guided/GuidedJsonLogitsProcessorTest.java
```

Use a tiny controlled model/vocabulary. Example schema:

```json
{
  "type": "object",
  "properties": {
    "foo": { "type": "integer" }
  },
  "required": ["foo"],
  "additionalProperties": false
}
```

Assert that after accepting prefix tokens for `{"foo":`, only integer-start tokens remain.

## Integration Test

Add a real-model test after unit tests pass.

Candidate:

```text
GemmaPromptIT.gemmaGuidedJsonTest
```

Keep the schema small and deterministic. Assert generated text:

- starts with `{`
- contains required property names
- parses as JSON
- satisfies the schema if a lightweight validator is available

## Guardrails

Reuse `SketchesSettings`.

The generated regex and index construction must obey:

- `maxRegexLength`
- `maxIndexStates`
- `maxIndexTransitions`

Error messages should clearly identify guided JSON as the source when possible, while preserving lower-level regex/index limit messages.

## Deferred

- Full `response_format` JSON-schema compatibility.
- Full JSON Schema draft coverage.
- CFG/BNF guided decoding.
- Rollback/backtracking in the generation loop.
- Performance optimizations and index caching.
