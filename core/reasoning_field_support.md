# Reasoning Field Support

Some modern chat models can emit a separate reasoning channel in addition to the final assistant answer. Deliverance preserves that split where the model/parser path exposes it, instead of forcing everything into normal response content.

## API Shape

Deliverance uses `reasoning_content` as the default chat API field name for streamed and non-streamed reasoning content.

The goal is to keep this distinction:

```text
assistant final answer -> content
assistant reasoning    -> reasoning_content
```

This matters for local agents and tools. Reasoning text can be displayed to the user or logged separately without being fed back into the next model turn as normal assistant content.

## Nanocode Behavior

`nanocode-deliverance` displays reasoning separately and avoids feeding reasoning text back into the conversation as assistant content.

That keeps follow-up prompts cleaner:

```text
assistant.content           -> part of conversation history
assistant.reasoning_content -> diagnostic/display channel
```

## vLLM Compatibility

The web/client path can opt into vLLM-style reasoning fields. This is useful when Deliverance fronts or interoperates with clients that already know about reasoning fields.

## Model Notes

Not every model emits a reasoning channel. Support is model- and prompt-template-dependent. For example, Gemma 4 documentation discusses `response.reasoning`, but enabling thinking does not guarantee visible reasoning for every prompt.

## Related Docs

- [Industry-standard chat API](chat_api.md)
- [Tool call parser](tool_parser.md)
- [Gemma 4 support](gemma4_support.md)
- [Nanocode Deliverance](../nanocode-deliverance/README.md)
