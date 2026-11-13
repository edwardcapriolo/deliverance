# Granite 4.1 8B LLMON Preview

## Selection

- Checkpoint: `ibm-granite/granite-4.1-8b-llmon-preview`
- Base model: `ibm-granite/granite-4.1-8b-base`
- Variant: instruction-tuned LLMON preview
- License: Apache-2.0
- Architecture: `GraniteForCausalLM`
- Config model type: `granite`
- Transformers version in config: `5.0.1.dev0`
- Release date in model card: November 12, 2026

The checkpoint is preferred over the base model because the goal is first-class LLMON support for an instruction model.

## Initial Finding

LLMON is a learned prompt representation and chat-template behavior, not a new transformer block according to the model card and post-training paper.

The model ships a `chat_template.jinja` that renders structured instruction, data, requirement, document, tool, and execution blocks into reserved special tokens.

The model card says conventional chat formatting is out of distribution.

The transformer architecture still requires independent verification against the upstream Transformers implementation.

## Phase Status

Phase 1 is in progress.

The exact `config.json` has been captured and every field is listed in `config-field-analysis.md`.

Implementation and source-test analysis have not started.

## References

- Model card: https://huggingface.co/ibm-granite/granite-4.1-8b-llmon-preview
- LLMON repository: https://github.com/generative-computing/llmon
- LLMON language paper: https://arxiv.org/abs/2603.22519
- LLMON post-training paper: https://arxiv.org/abs/2610.38120
- IBM Granite technical blog: https://huggingface.co/blog/ibm-granite/llmon-an-llm-native-markup-language-now-in-granite
- Base model: https://huggingface.co/ibm-granite/granite-4.1-8b-base
- Attention visualization: https://huggingface.co/spaces/ibm-granite/LLMON-Attention-Visualization
