# Industry-Standard Chat API

Deliverance exposes an industry-standard `/chat/completions` API from the `web` module. This API maps HTTP JSON requests onto the same embedded inference engine used by Java callers.

## Request Flow

The high-level flow is:

```text
HTTP /chat/completions
  -> generated OpenAPI request model
  -> ChatCompletionService.mapRequest(...)
  -> PromptSupport / chat template
  -> GeneratorParameters
  -> model.generate(...)
  -> chat completion response object or stream
```

The generated request model comes from:

```text
web/deliverance_specification.yaml
```

The request mapping lives in:

```text
web/src/main/java/net/deliverance/http/ChatCompletionService.java
```

## Generation Parameters

The web API maps common chat-completion fields into `GeneratorParameters`, including:

```text
temperature
top_p
max_tokens
seed
stop
logprobs
top_logprobs
guided_regex
guided_json
```

These are request-time controls. They do not change model weights or model constructors.

## Chat Templates

Deliverance uses model-specific prompt support to render chat messages into the format expected by the model.

```text
messages[]
  -> PromptSupport.Builder
  -> model chat template
  -> prompt string
  -> tokenizer
  -> generation loop
```

This is important because Llama, Gemma, Qwen, and other model families do not share one universal prompt format.

## Tool Calls

Tool requests are converted into Deliverance prompt-tool objects and then parsed from model output by model-family-specific tool parsers.

See [Tool call parser](tool_parser.md).

## Guided Output Fields

Deliverance exposes guided decoding fields directly on chat completion requests:

```json
{
  "guided_regex": "TICKET-[0-9]{4}"
}
```

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

See [Guided generation](guided_generation.md).

## Streaming

The controller supports streaming chat completions with server-sent events. Streaming tool-call deltas and reasoning fields are handled as API compatibility features where available.

## Related Docs

- [Inference engine flow](inference_flow.md)
- [Generator sampling](generator_sampling.md)
- [Guided generation](guided_generation.md)
- [Tool call parser](tool_parser.md)
- [Reasoning field support](reasoning_field_support.md)
