# Guided Regex Plan

## Goal

Add guided regex as a public generation constraint and use it as the foundation for future guided JSON support.

Guided regex is not just a user-facing feature. It is the internal mechanism needed for an Outlines-style path:

```text
JSON Schema -> regex -> automaton/guide -> allowed token ids -> logits mask
```

## Public API

- Add `GeneratorParameters.guidedRegex: Optional<String>`.
- Add `GeneratorParameters.withGuidedRegex(String regex)`.
- Reject conflicting guidance modes during request setup:
  - `guidedChoice` and `guidedRegex` cannot both be set.
  - Future `guidedJson` should join the same exclusivity rule.

## Web/OpenAPI

- Add `guided_regex` to `CreateChatCompletionRequest` in `web/deliverance_specification.yaml`.
- Regenerate OpenAPI sources through the existing web Maven generator.
- Map `request.getGuidedRegex()` in `ChatCompletionService.mapRequest(...)` to `params.withGuidedRegex(...)`.
- Add web mapping tests that assert the generated request field reaches `GeneratorParameters`.

## Core Adapter

Extend `LogitsProcessorFactory`:

```text
guidedChoice -> ChoiceGuide
guidedRegex  -> RegexGuide
```

Keep the adapter thin:

- Build a sketches `Vocabulary` from `AbstractModel`.
- Create a `Regex` term / `RegexGuide`.
- Wrap the guide in the existing `GuideLogitsProcessor`.

## sketches-core Runtime

Add `Vocabulary`:

- Forward map: decoded token string -> token IDs.
- Reverse map: token ID -> decoded token string.
- EOS token IDs.

Add `RegexGuide implements Guide`:

- Uses `Vocabulary`.
- Maintains accepted regex/automaton state.
- `getTokens()` returns valid next token IDs.
- `advance(tokenId)` updates state and rejects invalid token IDs.
- `isFinished()` indicates current state is accepting.

Add regex engine:

- Prefer a finite automaton implementation over Java `Pattern`, because guidance needs valid-next-token / prefix-state behavior, not final validation.
- If using a dependency like `dk.brics.automaton`, add it to `sketches-core`.
- If no dependency is used, port enough automaton behavior from outlines-core, but that is more work.

## Tests First

Port upstream `outlines-core` tests where relevant, even if some are disabled initially.

Start with enabled deterministic tests:

- `VocabularyTest`
- `RegexGuideTest`
- `GuidedRegexLogitsProcessorTest`

Add disabled/ported parity tests from upstream where the Java implementation is not ready.

## Initial Enabled Test Cases

### Simple Sequence

```text
regex = "ab"
vocab: a=1, b=2, c=3, EOS=0
```

Expected behavior:

- initial allowed: `1`
- after `1`: `2`
- after `2`: EOS

### Alternatives

```text
regex = "a(b|c)"
```

Expected behavior:

- after `a`: `b` and `c`

### Multi-Character Tokens

```text
regex = "ab"
vocab: a=1, b=2, ab=4, EOS=0
```

Expected behavior:

- initial allowed includes `1` and `4`

### Processor Masking

Given regex `ab` and generated token `a`, only token `b` keeps its logit. All other logits are set to `Float.NEGATIVE_INFINITY`.

### Conflict Validation

Using both `withGuidedChoice(...)` and `withGuidedRegex(...)` should throw a clear error.

## Deferred

- Full JSON schema support.
- Full OpenAI `response_format` JSON-schema semantics.
- CFG/BNF.
- Performance optimizations such as precomputed bitsets.
