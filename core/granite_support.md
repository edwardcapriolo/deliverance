# Granite 4.0 / GraniteMoeHybrid Support

Deliverance supports the Hugging Face `granitemoehybrid` model family used by IBM Granite 4.0 and Antares.

GraniteMoeHybrid is a family name, not one fixed block layout. A checkpoint can be dense attention-only, or it can combine Mamba sequence-modeling layers, transformer attention layers, and sparse Mixture-of-Experts feed-forward blocks.

## Supported Paths

### Dense Antares Path

`fdtn-ai/antares-1b` is a GraniteMoeHybrid checkpoint with all layers configured as attention and no active MoE experts:

```text
model_type=granitemoehybrid
num_local_experts=0
num_experts_per_tok=0
layer_types=["attention", ...]
```

Deliverance supports this path and verifies it with real-model smoke/parity tests. The important prompt/runtime detail is that Granite/Antares templates already own their special tokens, so Deliverance does not prepend an extra BOS token for this model family.

### Hybrid Granite Path

`ibm-granite/granite-4.0-h-tiny` is not tiny in local-runtime terms. It is a full hybrid model with roughly 6.9B total parameters and about 1B active parameters per token. Its config uses:

```text
layer_types=mamba plus periodic attention layers
num_local_experts=64
num_experts_per_tok=6
```

Deliverance supports loading and smoke-generating with this hybrid path. The current implementation includes:

- Granite attention layers using the existing causal attention path.
- Granite shared MLP formula.
- Granite sparse MoE feed-forward branch.
- Mamba slow-path inference.
- Tied output head fallback for checkpoints without `lm_head.weight`.

## Mamba, Attention, And MoE In Plain Terms

Transformer attention layers compare each token against previous tokens. They are flexible, but attention cost grows with context length.

Mamba layers are state-space sequence layers. They keep recurrent state and process sequences with a different memory/computation profile than attention. Granite 4.0 hybrid models use many Mamba layers and fewer attention layers.

MoE means Mixture of Experts. Instead of running one dense feed-forward network for every token, a router chooses a small number of expert feed-forward networks. Granite 4.0 hybrid tiny has many total expert weights, but only a subset are active per token.

## Prompt And Reasoning Notes

Granite/Antares chat templates render their own role and special-token markers. Deliverance has model-family tests for this because an extra BOS token caused bad Antares output.

Antares templates can end the rendered prompt inside a `<think>` block:

```text
<|start_of_role|>assistant<|end_of_role|><think>
```

That means generated text begins as reasoning until the model emits `</think>`. Deliverance represents this behavior in `ReasoningTextSplitter` tests so streaming code does not treat reasoning as normal assistant content.

## Quantization

Deliverance can quantize Antares-1B to JQ4. Local smoke testing showed coherent output from the generated `antares-1b-JQ4` model.

Granite 4.0 hybrid tiny can also be quantized, but the size reduction is limited today because its large MoE expert tensors are stored as three-dimensional tensors and the current quantizer keeps non-2D tensors dense.

## API Notes

Antares is trained as a terminal-style vulnerability-localization agent. Its official CLI expects a raw OpenAI-compatible `/v1/completions` endpoint, not chat completions. Deliverance includes a minimal legacy completions endpoint that passes raw prompt text directly to generation without applying a chat template.

## Tests

Important focused tests include:

- `GraniteMoeHybridHfTextModelPortedTest`: synthetic GraniteMoeHybrid block/layout/forward coverage.
- `GraniteMoeHybridSharedMlpTest`: shared MLP formula coverage.
- `ReasoningTextSplitterTest`: `<think>` reasoning/content splitting, including Antares prompts that already opened `<think>`.
- `LocalAntaresPromptTemplateTest`: cached Antares tool-template rendering.
- `AntaresFetchIT`: real Antares metadata/logits/generation smoke, tagged `large-model`.
- `GraniteTinyFetchIT`: real Granite 4.0 hybrid tiny metadata/load/generation smoke, tagged `large-model`.

## Current Limitations

- Mamba support is a correctness-oriented Java slow path, not a production-optimized kernel.
- Full HF parity for every GraniteMoeHybrid test is not claimed.
- Granite hybrid MoE quantization is limited by current Q4 support for 2D tensors.
- Antares agent-loop compatibility belongs with the `/v1/completions` path or the official Antares CLI, not generic chat tooling.
