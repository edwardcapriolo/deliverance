# Gemma4 Support

This page explains Deliverance's current Gemma 4 support at a high level, what works today, and how to use it.

## What Gemma4 Means In Deliverance

Gemma 4 is not treated as a simple Gemma 2 checkpoint refresh.

The current implementation includes Gemma 4 specific handling for:

- multimodal-style `text_config` parsing
- hybrid layer types:
  - `sliding_attention`
  - `full_attention`
- per-layer input embeddings (PLE)
- Gemma 4 chat templates and `enable_thinking`
- large logical tensors split into internal parts

Gemma 4 support is still evolving, but it is now far enough along to:

- fetch and load local Gemma 4 model assets
- render Gemma 4 chat prompts
- run generation with Gemma 4 specific model code
- use `grace` tokenizer behavior in the runtime path

## Current Scope

The main target today is:

- `google/gemma-4-E2B-it`

Deliverance also has infrastructure for broader Gemma 4 work, but E2B is the most actively exercised configuration.

## Using A Gemma4 Model

### Basic generation

```java
ModelFetcher fetch = new ModelFetcher("google", "gemma-4-E2B-it");
try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).build()) {
    PromptSupport.Builder prompt = model.promptSupport().orElseThrow().builder()
            .addUserMessage("What is the capital of the state of New York?");

    Response response = model.generate(
            UUID.randomUUID(),
            prompt.build(),
            new GeneratorParameters()
                    .withTemperature(0.0f)
                    .withMaxTokens(64),
            new DoNothingGenerateEvent()
    );

    System.out.println(response.responseText);
}
```

### Thinking mode

Gemma 4 prompt templates support a thinking toggle.

```java
PromptSupport.Builder prompt = model.promptSupport().orElseThrow().builder()
        .addTemplateArgs(Map.of("enable_thinking", true))
        .addUserMessage("Bob is a carpenter. Sara is a teacher. Who should you call to fix your roof?");
```

When the model emits a thought channel, Deliverance will parse it into:

- `response.responseText`
- `response.reasoning`

Note that enabling thinking does not guarantee that the model will always emit a visible reasoning channel for every prompt.

## Prompt And Tokenizer Path

For Gemma 4, the important thing is not only the rendered prompt string, but the exact prompt token ids.

Deliverance now supports using the `grace` tokenizer path in runtime encoding, which is important because Gemma 4 expects prompt-control tokens like:

- `<|turn>`
- `<turn|>`

to be tokenized correctly.

The legacy core tokenizer path can still be useful for comparison/debugging, but Gemma 4 runtime behavior should be validated against the `grace`/Hugging Face token ids.

## Local Models And Quantized Models

Deliverance stores fetched and local models under:

- `~/.deliverance/<owner>_<model>/`

This applies to:

- Hugging Face fetched models
- tokenizer-only fetches
- local quantized model outputs created by Deliverance

If the required local files already exist, the fetch path now prefers the local directory before contacting Hugging Face.

## Current Notes

Gemma 4 support is significantly improved, but the decoder path is still an active area of work.

When debugging Gemma 4 behavior, the most useful anchors are usually:

- Hugging Face prompt token ids
- focused unit tests around tokenizer behavior
- focused unit tests around KV cache reconstruction
- integration prompts with deterministic generation parameters

## Related Docs

- [Gemma4 Side-by-Side Analysis](../gemma4-side-by-side-analysis.md)
- [Inference engine flow](inference_flow.md)
- [Generator sampling](generator_sampling.md)
