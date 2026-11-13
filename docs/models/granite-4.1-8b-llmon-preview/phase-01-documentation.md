# Phase 1: Documentation And Model Selection

## Status

Complete for operator review. Do not begin source implementation analysis until approved.

## Selected Checkpoint

`ibm-granite/granite-4.1-8b-llmon-preview`

This is the instruction-tuned LLMON preview checkpoint, selected instead of the Granite 4.1 base checkpoint because the target capability is first-class LLMON support.

## Config Summary

- Architecture: `GraniteForCausalLM`
- Model type: `granite`
- 40 layers
- Hidden size: `4096`
- Intermediate size: `12800`
- Attention heads: `32`
- KV heads: `8`
- Context: `131072`
- Configured dtype: BF16
- Vocabulary: `100352`
- LLMON-specific config fields: none observed

## LLMON Summary

The model card, IBM technical blog, LLMON repository, and papers describe LLMON as an LLM-native markup and post-training format.

The checkpoint ships a chat template that renders structured instructions, data, requirements, documents, tools, tool results, and execution pointers using reserved tokens.

The model card states that conventional chat formatting is out of distribution.

The post-training paper states that the Granite transformer architecture is not changed; LLMON behavior is learned during post-training.

This means the likely port split is:

```text
Granite 4.1 transformer implementation
LLMON tokenizer/chat-template/prompt rendering
LLMON tool and structured-output integration
```

## Initial Risks

- The configured `transformers_version` is `5.0.1.dev0`; upstream source version must be pinned before implementation.
- `rope_parameters` and duplicate `rope_theta` precedence must be confirmed.
- `attention_multiplier`, `embedding_multiplier`, `residual_multiplier`, and `logits_scaling` must be mapped to exact forward locations.
- `attention_bias=false` and `mlp_bias=false` must be reflected in weight selection.
- `use_cache=false` needs explanation because Reckoning generation depends on KV cache behavior.
- LLMON requires exact tokenizer special-token and chat-template behavior; ordinary chat prompt rendering is not sufficient.
- The model is approximately 8B parameters and will require a deliberate CPU baseline and later quantization phase.

## Sources Reviewed

- Model card: https://huggingface.co/ibm-granite/granite-4.1-8b-llmon-preview
- Exact config: https://huggingface.co/ibm-granite/granite-4.1-8b-llmon-preview/raw/main/config.json
- LLMON repository: https://github.com/generative-computing/llmon
- LLMON language paper: https://arxiv.org/abs/2603.22519
- LLMON post-training paper: https://arxiv.org/abs/2610.38120
- IBM Granite blog: https://huggingface.co/blog/ibm-granite/llmon-an-llm-native-markup-language-now-in-granite
- Base model: https://huggingface.co/ibm-granite/granite-4.1-8b-base
- Attention visualization: https://huggingface.co/spaces/ibm-granite/LLMON-Attention-Visualization

## Next Phase After Approval

Inspect the complete model repository files, tokenizer/chat template, generation configuration, checkpoint index, Transformers source, and applicable tests.

## Initial LLMON Integration Plan

The first integration path will be client-side LLMON prompt composition sent to the existing raw completions endpoint.

```text
Java LLMON builder
  -> rendered LLMON prompt string
  -> /v1/completions prompt field
  -> server raw prompt path
  -> tokenizer
  -> inference
```

The server must not apply a normal chat template to an already-rendered LLMON prompt.

The initial Java helper should model instructions, data, requirements, documents, tools, tool results, execution bindings, and stable block IDs without requiring callers to write LLMON markup.

## Planned Tests

- Compare Java-rendered LLMON strings with the output of the model's Transformers `apply_chat_template` reference.
- Verify deterministic block ID generation and explicit `exec` bindings.
- Verify instruction, data, requirement, document, tool, and tool-result rendering.
- Verify `exec: false` content remains present but is not referenced by the execution block.
- Add a web integration test that sends the rendered prompt to `/v1/completions` and verifies tokenization/inference uses the raw prompt unchanged.
- Verify prompt token IDs against the Transformers reference.
- Add end-to-end LLMON tests for prompt injection separation and tool-call formatting.

Structured LLMON fields in `/v1/chat/completions` are a later API phase. The raw completions path avoids requiring web/client schema changes for the first usable implementation.
