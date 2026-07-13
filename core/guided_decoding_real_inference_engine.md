# Guided Decoding, Now It's A Real Inference Engine

## 1. What is guided decoding and why do you need it?

If you are familiar with Generative AI you likely know that the process is intensive. Large Language models 
have to run millions/billions/trillions of calculations over large arrays to pick the next token, and a token 
isn't always an entire word; sometimes it is only a part of the word.

An important part of the inference flow is the prefill. Prefill is the stage where the model processes the prompt 
before generating the first output token. You write instructions:

system: You are a helpful assistant.
user: Extract the person from this sentence: Alice is 42 years old and lives in Paris.

As a user you might want a compact JSON response like:

```json
{"name":"Alice","age":42,"city":"Paris"}
```

But the model may choose to be wordy: "Alice: 42-year-old Location: Paris, France, located in Europe. Known for art and wine." You can try to constrain it 
with more prompting, such as "Reply only with valid JSON matching this schema." However, that has downsides: the prompt is bigger so the prefill will take longer, and the model might not understand your direction and still answer in a way you don't expect.

Guided decoding is one of the things that makes the inference engine a "real" inference engine. If you want to use 
an inference engine in a programmatic workflow like an agent framework, the components typically need to 
exchange messages matching exact schema. If they want JSON in a specific format, a run-on blob of text won't work.

Deliverance now has support for the guided decoding options: Guided Choice, Guided Regex, and one 
of the most powerful Guided JSON.  


## 2. Guided Choice First Implementation

The first guided decoding feature in Deliverance was guided choice.

The idea is simple: if the user says the answer must be one of a small set of strings, the sampler should only choose tokens that can still lead to one of those strings.

For example:

```java
new GeneratorParameters()
        .withGuidedChoice(List.of("Giants", "Jets"))
```

If the model has generated nothing yet, the next token must be a valid first token of either `Giants` or `Jets`. If it has already generated the token prefix for `Gi`, then the next token must keep that prefix on a path toward `Giants`.

The first version of this lived in a special `GuidedChoiceSampler`. That worked, but it was not the right long-term shape. Guided decoding is not a separate kind of model sampling. It is a constraint applied to the normal sampling flow.

The improved version turns guided choice into a guide-backed logits processor:

```text
generated state
  -> guide says which token ids are allowed next
  -> logits processor masks everything else
  -> normal sampler picks from remaining tokens
```

That was the important shift. Guided choice became the first real guided decoding path instead of a one-off sampler fork.

## 3. What Are FSAs?

FSA means finite-state automaton. FSAs are small state machines that tell us which characters or tokens can come next.

Consider this regex:

```text
a(b|c)
```

This means:

```text
first:  a
then:   b or c
```

As a state machine:

```text
state 0 --a--> state 1
state 1 --b--> final
state 1 --c--> final
```

So after the model has generated `a`, the guide knows that only `b` or `c` can come next.

That is the bridge to guided regex. A regex becomes an automaton. The automaton becomes token transitions. The sampler uses those transitions to decide which token ids are allowed.

Deliverance's flow is:

```text
regex + Vocabulary
  -> Index
  -> IndexGuide
  -> GuideLogitsProcessor
```

`Vocabulary` bridges model tokens and text:

```text
token id -> decoded token text
decoded token text -> token ids
```

`Index` asks: from this automaton state, if I append this token's decoded text, do I land in a valid next state?

That is how regular expressions become token-level decoding constraints.

## 4. Negative Infinity Sounds Really Cool

Models do not directly output words. They output logits: scores for every possible next token.

Imagine the model has four possible next tokens:

```text
cat    score 10
dog    score 20
ma     score 30
zebra  score 40
```

Normally the sampler might pick `zebra`, because it has the highest score.

But suppose the guided target is `dogma`, and the model has already generated `dog`. The only token that keeps the output on track is `ma`.

So guided decoding changes the scores like this:

```text
cat    -infinity
dog    -infinity
ma     30
zebra  -infinity
```

Now the sampler can still run normally. It does not need to know about JSON, regex, or choices. It just sees that `ma` is the only viable token.

This is the core trick:

```text
do not force the sampler to understand the constraint
make impossible tokens impossible by setting their logits to negative infinity
```

That is why guided decoding composes cleanly with normal inference. The model still produces logits. The sampler still samples. The guide only edits the logits before sampling.

## 5. JSON Schema

Guided JSON builds on guided regex.

The stack is:

```text
JSON Schema
  -> regex
  -> finite-state automaton
  -> token Index
  -> Guide
  -> logits mask
```

For example, this schema:

```json
{
  "type": "object",
  "properties": {
    "name": { "const": "Alice" },
    "age": { "type": "integer" }
  },
  "required": ["name", "age"],
  "additionalProperties": false
}
```

can become a regex that describes JSON like:

```json
{"name":"Alice","age":42}
```

Once the schema is a regex, the rest of the system does not need a special JSON sampler. It reuses the same regex/index/guide machinery.

This matters because JSON validation after generation is too late. A validator can say the final output is invalid. Guided decoding prevents invalid tokens from being selected in the first place.

Current guided JSON support handles practical schema pieces such as objects, arrays, `enum`, `const`, `anyOf`, `oneOf`, integers, numbers, booleans, nulls, and basic strings. Broader JSON Schema support is still expanding, and the path is intentionally incremental and test-driven.

## 6. The Inference Benchmark

The guided benchmark exists and runs:

```sh
sh benchmarks/run-qwen-guided-benchmark.sh
```

It uses a separate suite:

```text
benchmarks/guided-support-suite.jsonl
```

The suite covers:

```text
guided choice
guided regex
guided JSON
small schemas
larger schemas
polymorphic cat/dog style schemas
```

The inference profiler now shows guided decoding costs directly. The benchmark showed the important cost pattern: guided decoding runtime is mostly normal inference, but setup can matter. That is why Deliverance caches JSON-schema-to-regex and per-model regex indexes.

For the technical metric names and profiling rows, see [Guided generation](guided_generation.md).

This is no longer a toy feature. We can run it against quantized Qwen3 and see where the time goes.

## 7. How Far We Have Come

Deliverance did not start with the standard inference-engine feature set. It grew into it.

There was a time when generation was basically greedy sampling. Now the sampler supports the features users expect from serious inference engines:

```text
temperature
top-p / nucleus sampling
top-k
logprobs
top logprobs
stop strings
max token controls
XTC / exclude top choice
guided choice
guided regex
guided JSON
```

See [Generator sampling](generator_sampling.md) for temperature, top-p / nucleus sampling, top-k, logprobs, top logprobs, and XTC / exclude top choice.

That matters because modern inference engines are not just matrix multiplication loops. They are request processors. The center of that in Deliverance is `GeneratorParameters`: the object that collects the knobs a real application needs for one generation request.

`GeneratorParameters` now covers the kind of controls people expect from a serious local inference engine:

```java
new GeneratorParameters()
        .withTemperature(0.0f)
        .withTopK(64)
        .withTopP(0.95f)
        .withLogProbs(true)
        .withTopLogProbs(5)
        .withGuidedRegex("TICKET-[0-9]{4}");
```

Add the industry-standard chat API, tool calls, reasoning fields, streaming, and guided JSON on top of that, and it starts to feel a lot less like a toy local runner and a lot more like the core feature set of the major inference engines. Holy crap, Deliverance is getting pretty close.

Guided BNF / CFG is still missing, but the current `Guide` / `Index` structure is the place it will fit.

Related feature docs:

- [Industry-standard chat API](chat_api.md)
- [Generator sampling](generator_sampling.md)
- [Guided generation](guided_generation.md)
- [Tool call parser](tool_parser.md)
- [Reasoning field support](reasoning_field_support.md)

The goal is not only to chase giant-server inference stacks. Deliverance is optimized for the common case: local hardware, practical quantized models, CPU-first execution, and users who want control without a massive serving platform.

Guided decoding is another step in that direction. It turns Deliverance from "a thing that can run a model" into a more complete inference engine: one that can sample flexibly, shape outputs, measure costs, and expose modern structured-generation features on normal hardware.
