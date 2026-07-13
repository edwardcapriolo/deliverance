# Guided Generation

Deliverance supports guided generation: request-time constraints that limit which tokens the sampler may choose. This is how features such as `guided_choice`, `guided_regex`, and `guided_json` force structured outputs during generation instead of validating and repairing text after the model has already produced it.

## Features

### Guided Choice

Guided choice constrains output to exactly one of a fixed set of strings.

```java
Response response = model.generate(
        UUID.randomUUID(),
        prompt.build(),
        new GeneratorParameters()
                .withTemperature(0.0f)
                .withGuidedChoice(List.of("Giants", "Jets")),
        new DoNothingGenerateEvent());
```

This existed before, but it used a separate `GuidedChoiceSampler` path. It now goes through the same logits processor lifecycle as other guided modes. That means guided choice benefits from the normal sampler path and shares the same request setup, processing, and acceptance flow as regex and JSON guidance.

### Guided Regex

Guided regex constrains output to match a regular expression.

```java
Response response = model.generate(
        UUID.randomUUID(),
        prompt.build(),
        new GeneratorParameters()
                .withTemperature(0.0f)
                .withGuidedRegex("TICKET-[0-9]{4}"),
        new DoNothingGenerateEvent());
```

The chat API exposes this as `guided_regex`, following the vLLM-style request field:

```json
{
  "model": "qwen3",
  "messages": [
    { "role": "user", "content": "Create a support ticket id." }
  ],
  "guided_regex": "TICKET-[0-9]{4}",
  "temperature": 0,
  "max_tokens": 32
}
```

### Guided JSON

Guided JSON constrains output to a JSON Schema. The schema is compiled to a regular expression and then to the same finite-state/token guide used by guided regex.

```java
String schema = """
        {
          "type": "object",
          "properties": {
            "name": { "const": "Alice" },
            "age": { "type": "integer" },
            "city": { "const": "Paris" }
          },
          "required": ["name", "age", "city"],
          "additionalProperties": false
        }
        """;

Response response = model.generate(
        UUID.randomUUID(),
        prompt.build(),
        new GeneratorParameters()
                .withTemperature(0.0f)
                .withGuidedJson(schema),
        new DoNothingGenerateEvent());
```

The web API exposes this as `guided_json`:

```json
{
  "model": "qwen3",
  "messages": [
    { "role": "user", "content": "Extract the person from: Alice is 42 and lives in Paris." }
  ],
  "guided_json": {
    "type": "object",
    "properties": {
      "name": { "const": "Alice" },
      "age": { "type": "integer" },
      "city": { "const": "Paris" }
    },
    "required": ["name", "age", "city"],
    "additionalProperties": false
  },
  "temperature": 0,
  "max_tokens": 64
}
```

Current JSON Schema support includes the subset needed for useful structured extraction and benchmarking:

- `type: object`
- `properties`
- `required` in deterministic schema order
- `string`
- `integer`
- `number`
- `boolean`
- `null`
- arrays with `items`
- `enum`
- `const`
- `anyOf`
- `oneOf`

Object property permutations are intentionally not expanded. JSON objects are semantically unordered, but guided generation only needs one valid output ordering. Deliverance currently emits fields in schema order to avoid factorial regex and index growth.

## Relationship To Sketches And Outlines

The `sketches-core` module contains the structured generation model and runtime pieces inspired by Outlines / outlines-core:

- `Term` types such as `Choice`, `Regex`, and `JsonSchema`
- `Vocabulary`
- `Index`
- `Guide`
- `IndexGuide`
- `JsonSchemaRegexBuilder`

Outlines separates user-facing structured output APIs from lower-level guided decoding machinery. Its lower-level `outlines-core` library uses concepts such as `Vocabulary`, `Index`, and `Guide` to map text constraints to allowed token ids. Deliverance follows the same broad architecture with Java-specific integration:

```text
User request
  -> GeneratorParameters
  -> sketches-core term/regex/index/guide
  -> core LogitsProcessor
  -> sampler masks invalid logits
```

The split is intentional:

- `core` owns inference, model execution, sampling, and web-facing request parameters.
- `sketches-core` owns structured-output guide mechanics.
- The core adapter translates Deliverance models/tokenizers into sketches `Vocabulary` and wraps guides as logits processors.

## Generation Flow

The generation loop now creates at most one `LogitsProcessor` per request:

```text
GeneratorParameters
  -> LogitsProcessorFactory
  -> GuideLogitsProcessor
```

The sampler lifecycle is:

```text
model forward
  -> output projection produces logits
  -> model logit transforms
  -> logitsProcessor.process(logits, responseContext)
  -> normal argmax/top-k/top-p sampling
  -> logitsProcessor.accept(chosenToken, responseContext)
  -> generation loop appends token to ResponseContext
```

`process(...)` masks invalid tokens by setting their logits to `Float.NEGATIVE_INFINITY`. `accept(...)` advances guide state after the sampler chooses a token. The guide is request-scoped and mutable; the `Index` is immutable and can be cached.

Only one guided mode may be active for a request:

```text
guided_choice
guided_regex
guided_json
```

## Finite-State Guidance

Guided regex and guided JSON use finite-state automata.

The path is:

```text
regex + Vocabulary
  -> Index
  -> state -> token id -> next state transitions
  -> IndexGuide
```

`Vocabulary` is not the tokenizer. It is a bridge between decoded token text and token ids:

```text
decoded token text -> token ids
token id -> decoded token text
EOS token ids
```

`Index` precomputes valid token transitions by walking each decoded token string through the regex automaton from each reachable state. At runtime, `IndexGuide.getTokens()` is a cheap lookup of allowed token ids for the current state.

## JSON Schema To Regex

Guided JSON uses:

```text
JSON Schema -> regex -> Index -> Guide
```

This is not because regex is a better JSON validator. It is because generation needs prefix-state behavior:

```text
Given the current partial output, which token ids can still lead to valid output?
```

Regular validators usually answer only whether a complete JSON document is valid. The automaton/index path answers the next-token question needed by the sampler.

## Safety Limits

Regex and JSON guidance can create large automata or large token indexes. Server-side limits live in `SketchesSettings`, not request parameters:

```java
new SketchesSettings(maxRegexLength, maxIndexStates, maxIndexTransitions)
```

`AutoModelForCausaLm.Builder` exposes:

```java
withSketchesSettings(SketchesSettings settings)
```

These limits protect the service from expensive user-supplied regexes or schemas. End users submit guidance constraints; operators configure limits.

## Metrics And Profiling

Guided generation reports profiling metrics through the existing `InferenceProfiler` and Dropwizard registry. Useful rows include:

- `guided.factory`
- `guided.vocabulary_build`
- `guided.json_schema_to_regex`
- `guided.index_build`
- `guided.logits_process`
- `guided.accept`
- `guided.regex.length`
- `guided.index.states`
- `guided.index.transitions`
- `guided.allowed_tokens`
- `guided.masked_tokens`
- `guided.index_cache.hit` / `guided.index_cache.miss`
- `guided.json_regex_cache.hit` / `guided.json_regex_cache.miss`

Run benchmarks with `--profile-stages` to see guided setup and per-token masking costs.

## Benchmark

A guided support benchmark suite is available:

```text
benchmarks/guided-support-suite.jsonl
```

Run it against Qwen3 JQ4:

```sh
sh benchmarks/run-qwen-guided-benchmark.sh
```

Outputs:

```text
core/target/qwen-guided-benchmark.csv
core/target/qwen-guided-benchmark.jsonl
```

The suite includes guided choice, guided regex, and multiple guided JSON schemas, including polymorphic and larger object examples.
