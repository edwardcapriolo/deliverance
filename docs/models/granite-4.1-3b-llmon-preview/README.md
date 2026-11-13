# Granite 4.1 3B LLMON Preview

## Selection

- Checkpoint: `ibm-granite/granite-4.1-3b-llmon-preview`
- Base model: `ibm-granite/granite-4.1-3b-base`
- Variant: instruction-tuned LLMON preview
- License: Apache-2.0
- Architecture: `GraniteForCausalLM`
- Config model type: `granite`
- Transformers version in config: `5.0.1.dev0`
- BF16 parameters: approximately 3.66B

This is the active port target. The 8B dossier remains a deferred follow-up validation target.

## What LLMON Is

LLMON is a learned prompt representation and chat-template behavior, not a new transformer block according to the model card and post-training paper.

It is an LLM-native markup language for separating authoritative instructions from material supplied as data. Its structured blocks can represent instructions, data, requirements, documents, tools, tool results, assistant tool calls, and execution bindings. The Granite checkpoint ships a chat template that turns those structures into reserved-token text the model was trained to consume.

The model card states that conventional chat formatting is out of distribution.

The Java entry point is `LlmOnPrompt`. It lets callers build these structures with typed builders and render the exact prompt text needed by the raw completion request, rather than hand-writing LLMON markup or JSON. This is a client-side prompt composition layer; it does not add a separate model architecture.

## Learn More

Start with the [LLMON language paper](https://arxiv.org/abs/2603.22519) for the language design and the [LLMON repository](https://github.com/generative-computing/llmon) for the reference renderer and examples. For Granite-specific usage and training context, see the [LLMON post-training paper](https://arxiv.org/abs/2610.38120) and the [IBM Granite explainer](https://huggingface.co/blog/ibm-granite/llmon-an-llm-native-markup-language-now-in-granite).

## References

- Model card: https://huggingface.co/ibm-granite/granite-4.1-3b-llmon-preview
- Exact config: https://huggingface.co/ibm-granite/granite-4.1-3b-llmon-preview/raw/main/config.json
- LLMON repository: https://github.com/generative-computing/llmon
- LLMON language paper: https://arxiv.org/abs/2603.22519
- LLMON post-training paper: https://arxiv.org/abs/2610.38120
- IBM Granite technical blog: https://huggingface.co/blog/ibm-granite/llmon-an-llm-native-markup-language-now-in-granite
- Base model: https://huggingface.co/ibm-granite/granite-4.1-3b-base
- Attention visualization: https://huggingface.co/spaces/ibm-granite/LLMON-Attention-Visualization
