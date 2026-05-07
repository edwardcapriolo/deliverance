# Gemma4 Side-by-Side Analysis

This document compares the current Deliverance `gemma4` implementation against upstream and adjacent runtimes before further implementation work.

Primary goals:
- identify the exact Gemma4 text-path steps
- map those steps to the current Deliverance code
- call out confirmed mismatches and likely causes of slow/incorrect behavior
- separate architecture facts from serving/runtime packaging facts

## References

- Deliverance current code
  - `core/src/main/java/io/teknek/deliverance/model/gemma4/Gemma4Model.java`
  - `core/src/main/java/io/teknek/deliverance/generator/Gemma4CausalSelfAttention.java`
  - `core/src/main/java/io/teknek/deliverance/generator/Gemma4TransformerBlock.java`
  - `core/src/main/java/io/teknek/deliverance/model/gemma4/Gemma4Config.java`
  - `safetensors/src/main/java/io/teknek/deliverance/safetensors/DefaultWeightLoader.java`
  - `safetensors/src/main/java/io/teknek/deliverance/safetensors/Weights.java`
- Hugging Face upstream
  - `transformers/models/gemma4/modeling_gemma4.py`
  - `transformers/models/gemma4/configuration_gemma4.py`
- llama.cpp
  - `convert_hf_to_gguf.py`
  - `src/llama-model.cpp`
- Ollama
  - Gemma family library/runtime packaging pages

## Bottom Line

The current Deliverance Gemma4 path is not yet a faithful implementation of upstream Gemma4 text inference.

The two biggest confirmed issues are:

1. attention math is not yet equivalent to upstream
2. the current attention implementation is dramatically slower than acceptable on CPU

The current code has useful scaffolding:

- nested `text_config` parsing
- Gemma4 layer metadata parsing
- prompt/template handling
- split-tensor loader improvements for large logical tensors
- a first-pass PLE pipeline

But the core decoder path still has correctness gaps.

## What Hugging Face Actually Does

### Decoder Layer Order

Upstream `Gemma4TextDecoderLayer.forward` is:

1. `input_layernorm`
2. `self_attn`
3. `post_attention_layernorm`
4. residual add
5. `pre_feedforward_layernorm`
6. `mlp`
7. optional MoE branch handling
8. `post_feedforward_layernorm`
9. residual add
10. optional PLE injection:
    - `per_layer_input_gate`
    - activation
    - elementwise multiply by `per_layer_input`
    - `per_layer_projection`
    - `post_per_layer_input_norm`
    - residual add
11. multiply by `layer_scalar`

### PLE Construction

Upstream constructs PLE in two parts:

1. token identity component
   - `embed_tokens_per_layer(input_ids)`
   - reshaped to `[B, S, num_hidden_layers, hidden_size_per_layer_input]`

2. context-aware projection component
   - `per_layer_model_projection(inputs_embeds)`
   - scaled by `hidden_size**-0.5`
   - reshaped per layer
   - normalized by `per_layer_projection_norm`

Then upstream combines them as:

`(per_layer_projection + per_layer_inputs) * (1 / sqrt(2))`

Each layer receives `per_layer_inputs[:, :, i, :]`.

### Attention Facts

Layer type comes from `config.layer_types[i]`:

- `sliding_attention`
- `full_attention`

Attention differences:

- sliding layers use sliding-window causal masks
- full layers use full causal masks
- full layers may use `global_head_dim`
- full layers may use `attention_k_eq_v`
- RoPE parameters are per layer type, not globally uniform

Projection shapes from upstream:

#### Sliding attention

- `q_proj`: `num_attention_heads * head_dim`
- `k_proj`: `num_key_value_heads * head_dim`
- `v_proj`: `num_key_value_heads * head_dim`
- `o_proj` input: `num_attention_heads * head_dim`

#### Full attention

- `q_proj`: `num_attention_heads * global_head_dim`
- `k_proj`:
  - if `attention_k_eq_v == false`: `num_key_value_heads * global_head_dim`
  - if `attention_k_eq_v == true`: `num_global_key_value_heads * global_head_dim`
- `v_proj`:
  - omitted when `attention_k_eq_v == true`
  - otherwise same output width as `k_proj`
- `o_proj` input: `num_attention_heads * global_head_dim`

Norm and RoPE order upstream:

- `q_proj` -> `q_norm` -> RoPE
- `k_proj` -> `k_norm` -> RoPE
- `v_proj` -> `v_norm`

Attention scores are scaled by `head_dim^-0.5` before softmax.

### Shared KV Upstream

Upstream shared-KV behavior is keyed by `layer_type`, not by arbitrary earlier layer index bookkeeping.

- non-shared layers compute/store full-length KV for later same-type shared layers
- shared layers reuse those stored same-type KV tensors
- this is separate from the normal `past_key_values` path because sliding-window cache state may not hold full-length KV

### Final Output Upstream

- final `norm`
- tied `lm_head` to `embed_tokens.weight`
- optional final logit softcap

## What Deliverance Currently Does

### High-Level Path

Current Deliverance Gemma4 path:

1. `Gemma4Model.loadInputWeights()` loads `embed_tokens.weight`
2. `batchForward(...)`
   - builds embeddings
   - computes packed PLE tensor
   - runs all decoder layers
3. `Gemma4TransformerBlock.forward(...)`
   - attention
   - FF
   - per-layer input gate/projection/norm
   - layer scalar
4. `loadOutputWeights()`
   - uses final norm if found
   - otherwise fallback identity norm
   - ties to embedding weights if no explicit `lm_head`

### PLE in Deliverance

Current Deliverance PLE path is conceptually close to upstream:

- logical row load from `embed_tokens_per_layer.weight`
- packed global `per_layer_model_projection`
- packed global `per_layer_projection_norm`
- combine with `1/sqrt(2)` scale
- per-layer gating/projection/norm in `Gemma4TransformerBlock`

This is the most promising part of the current implementation.

### Loader Changes Already Made

`DefaultWeightLoader` now exposes a generic row-sliced load path:

- `WeightLoader.loadRows(name, rowOffset, rowCount)`

This is important because very large logical 2D tensors can be internally split into synthetic `-part-*` segments to avoid Java mmap/int-size limits.

This is the right abstraction boundary:

- split tensors are a loader concern
- models should still ask for the logical tensor

This matches llama.cpp more closely than the earlier Gemma4-specific `-part-*` hacks.

## What llama.cpp Contributes

llama.cpp is useful here for format/runtime design, not as the exact decoder-math oracle.

Confirmed relevant facts:

- there is a dedicated Gemma4 architecture path
- the converter/runtime explicitly understands per-layer token embeddings
- it treats large model files and split output generically
- it carries Gemma4 metadata for:
  - sliding-window behavior
  - shared KV layers
  - per-layer embedding input length
  - Gemma4 size types like `E2B`, `E4B`, `26B.A4B`

Most important takeaway from llama.cpp:

- large-tensor or shard handling belongs in the loader/converter/runtime layer
- it should not be a Gemma4-specific special case in the model implementation

## What Ollama Contributes

Ollama is not an architectural oracle for Gemma4 math.

It is still useful for:

- packaging expectations
- template/runtime conventions
- deployable model bundle expectations
- practical serving behavior

It is not useful for:

- exact decoder-layer math
- exact RoPE logic
- exact KV-sharing semantics
- exact projection widths and normalization order

So Ollama should only influence:

- prompt/template handling
- model packaging assumptions

not decoder implementation.

## Confirmed Mismatches

### 1. Attention score scaling is missing

Upstream scales attention logits by `head_dim^-0.5`.

Current Deliverance `Gemma4CausalSelfAttention.score(...)` does not apply that scaling before softmax.

This alone can destroy generation quality.

### 2. Full-attention KV sizing is likely wrong

Current `Gemma4Config.getLayerKeyValueHeads("full_attention")` prefers `numGlobalKeyValueHeads` whenever present.

Upstream full-attention behavior is more specific:

- when `attention_k_eq_v == false`, full layers still use `num_key_value_heads`
- `num_global_key_value_heads` matters in the `attention_k_eq_v` path

Current Deliverance likely over-applies `numGlobalKeyValueHeads` for full attention.

### 3. `attention_k_eq_v` is not properly implemented

Current loader always expects distinct `k_proj` and `v_proj` for non-shared layers.

Upstream allows:

- `v_proj = None`
- values reused from keys when `attention_k_eq_v == true`

This is not just a future edge case. It is part of the Gemma4 architecture surface.

### 4. RoPE implementation is not upstream-equivalent

Current Deliverance precomputes per-layer-type frequencies from:

- `rope_theta`
- `partial_rotary_factor`

But upstream uses:

- `ROPE_INIT_FUNCTIONS`
- layer-type-specific rope init
- `proportional` rope handling
- additional `attention_scaling`
- dynamic rope update support

Current Deliverance does not model the full upstream rope behavior, especially for full-attention proportional rope.

This is a major likely correctness bug.

### 5. Masking behavior is oversimplified

Current Deliverance uses manual causal/window skipping in the attention loop.

Upstream explicitly constructs masks:

- full causal mask
- sliding-window causal mask
- optional bidirectional/multimodal-aware variants

For a text-only tiny prompt, the oversimplification may still run, but it is not equivalent to upstream and will become more wrong as complexity increases.

### 6. Shared-KV path is only approximate

Current Deliverance uses thread-local copied tensors keyed by source layer index.

Upstream behavior is keyed by layer type and explicitly stores full-length same-type KV states for later shared layers.

The current approach may be directionally similar, but it is not derived closely enough from upstream semantics to trust yet.

### 7. Attention implementation is scalar and extremely slow

Current `Gemma4CausalSelfAttention.forward(...)` does nested Java loops over:

- prompt position
- head
- prior token positions
- head dimension

and uses repeated scalar `get(...)` / `set(...)`.

This is a strong explanation for:

- ~160 second time-to-first-token on a tiny prompt
- severe CPU inefficiency compared with existing Gemma2 path

Even if correct, this implementation is too slow.

### 8. Current code still mixes architecture work and fallback heuristics

Examples:

- `IdentityLayerNorm` fallback for missing final norm
- root-resolution heuristics for multimodal text weights
- generic fallback assumptions for output head

These are survivable compatibility aids, but they should not be confused with a validated Gemma4 decoder implementation.

## Things That Look Reasonable So Far

These pieces are not yet fully validated, but they are directionally plausible:

- nested `text_config` parsing
- generic row-sliced logical tensor loading for large split tensors
- use of packed global PLE table + packed global PLE projection
- layer-local `per_layer_input_gate` / `per_layer_projection` / `post_per_layer_input_norm`
- tied embedding fallback for output head
- `chat_template.jinja` fallback support in tokenizer/template fetch/load path

## Why the Current Output is Both Slow and Wrong

The current observed behavior:

- tiny prompt
- very long TTFT
- gibberish text

fits the current mismatch list well.

The most likely explanations are:

1. severe attention inefficiency from scalar nested loops
2. at least one major math mismatch in attention:
   - missing scaling
   - wrong full-attention KV sizing
   - incomplete RoPE equivalence
   - approximate shared-KV semantics

This means profiling alone is not enough.

## Recommended Next Steps

### 1. Freeze new Gemma4 behavior work until the math diff is reduced

Do not add more architectural behavior before fixing the known mismatches above.

### 2. Fix the obvious attention mismatches first

In order:

1. apply correct attention score scaling
2. correct full-attention KV sizing rules
3. implement `attention_k_eq_v` properly
4. replace current rope logic with a closer upstream-equivalent implementation

### 3. Keep loader-side split tensor handling generic

The `loadRows(...)` direction is the right design.

Avoid reintroducing Gemma4-specific `-part-*` logic into model code.

### 4. Rework attention implementation for performance

After correctness fixes, the current attention path still needs a performance redesign.

It should move closer to:

- batched/provider-backed dot products
- less scalar looping
- fewer tensor page traversals per token/head

### 5. Only then do tensor-by-tensor validation if still needed

Activation dumping is still useful, but only after the code path is brought much closer to upstream.

At the current stage, the side-by-side code mismatch list is the more important tool.

## Short Checklist

- [ ] fix attention logit scaling
- [ ] fix full-attention KV head sizing rules
- [ ] implement `attention_k_eq_v`
- [ ] implement upstream-equivalent full/sliding RoPE handling
- [ ] validate shared-KV semantics against upstream same-type storage behavior
- [ ] redesign attention for provider-backed math instead of scalar loops
- [ ] add regression tests for logical row slicing of internally split tensors

## Final Assessment

The original implementation direction was not grounded tightly enough in a side-by-side execution diff.

The right path now is:

1. use this mismatch list as the implementation backlog
2. fix the mathematically obvious gaps first
3. keep loader concerns generic
4. only then return to runtime validation and performance tuning
