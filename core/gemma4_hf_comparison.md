# Gemma4 Hugging Face Comparison

Date: 2026-06-19

Compared against Hugging Face `transformers` main branch:

- `src/transformers/models/gemma4/modeling_gemma4.py`
- `src/transformers/models/gemma4/configuration_gemma4.py`

Local Deliverance files compared:

- `core/src/main/java/io/teknek/deliverance/model/gemma4/Gemma4Config.java`
- `core/src/main/java/io/teknek/deliverance/model/gemma4/Gemma4Model.java`
- `core/src/main/java/io/teknek/deliverance/generator/Gemma4CausalSelfAttention.java`
- `core/src/main/java/io/teknek/deliverance/generator/Gemma4TransformerBlock.java`
- `core/src/main/java/io/teknek/deliverance/model/gemma4/Gemma4PleSupport.java`
- `core/src/main/java/io/teknek/deliverance/generator/Gemma4RmsNormSupport.java`

## Important Correction

Gemma4 is not Gemma3n.

The actual checkpoint metadata says:

```json
"architectures": ["Gemma4ForConditionalGeneration"],
"model_type": "gemma4"
```

The Hugging Face Gemma4 text path does not include Gemma3n AltUp, Laurel, or activation sparsity. Those features should not be added to Deliverance Gemma4 unless the actual Gemma4 checkpoint contains those fields/weights.

## HF Gemma4 Text Architecture Summary

### Config

HF `Gemma4TextConfig` includes:

```python
vocab_size = 262_144
hidden_size = 2304
intermediate_size = 9216
num_hidden_layers = 30
num_attention_heads = 8
num_key_value_heads = 4
head_dim = 256
sliding_window = 512
final_logit_softcapping = None
use_bidirectional_attention = None
vocab_size_per_layer_input = 262_144
hidden_size_per_layer_input = 256
num_global_key_value_heads = None
global_head_dim = 512
attention_k_eq_v = False
num_kv_shared_layers = 0
enable_moe_block = False
use_double_wide_mlp = False
```

Default behavior:

```python
if self.layer_types is None:
    sliding_window_pattern = 6
    self.layer_types = [
        "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
        for i in range(self.num_hidden_layers)
    ]

if self.layer_types and self.layer_types[-1] != "full_attention":
    self.layer_types[-1] = "full_attention"

if self.rope_parameters is None:
    self.rope_parameters = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0},
    }
```

Deliverance currently assumes `layer_types` and `rope_parameters` are present. The checked-in fixture `core/src/test/resources/gemma_4_config.json` does provide both, so this is not currently the likely failure for that fixture. It is still a robustness difference.

### MLP

HF `Gemma4TextMLP`:

```python
first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer
self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)

down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

Deliverance equivalent:

- `Gemma4Config.getLayerHiddenLength(...)`, lines 165-168
- `VariableMLPBlock.forward(...)`, lines 69-99
- `Gemma4Model.loadTransformerBlockWeights(...)`, lines 164-172

Assessment: matches the main dense Gemma4 MLP structure after removing the mistaken Gemma3n activation-sparsity wiring from Gemma4.

### Text Attention Scaling And Softcap

HF `Gemma4TextAttention` sets:

```python
self.scaling = 1.0
```

HF eager attention:

```python
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
```

HF Gemma4 text attention does not pass an attention-logit softcap. It only uses final logit softcap at the LM head.

Deliverance equivalent:

- `Gemma4CausalSelfAttention.score(...)`, lines 388-397

Deliverance score is an unscaled dot product, which is equivalent to HF scaling `1.0`.

Deliverance only applies attention softcap if `config.attnLogitSoftCapping != null`. The Gemma4 fixture has `final_logit_softcapping` but no text `attn_logit_softcapping`, so attention softcap should not activate for Gemma4 text.

Assessment: attention score scaling is not the likely mismatch.

### Q/K/V Projection And Norm

HF `Gemma4TextAttention`:

```python
query_states = self.q_proj(hidden_states).view(hidden_shape)
query_states = self.q_norm(query_states)
query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
query_states = query_states.transpose(1, 2)

if self.is_kv_shared_layer:
    key_states, value_states = shared_kv_states[self.layer_type]
else:
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

    key_states = self.k_norm(key_states)
    key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
    key_states = key_states.transpose(1, 2)

    value_states = self.v_norm(value_states)
    value_states = value_states.transpose(1, 2)
```

Deliverance equivalent:

- Q projection/norm/RoPE: `Gemma4CausalSelfAttention.forward(...)`, lines 113-118 and 134-140
- K/V projection/norm/RoPE: lines 119-129 and 134-140
- V norm without scale: `Gemma4RmsNormSupport.applyInPlace(..., weights=null)`, line 128

Assessment: broadly matches. One important conditional remains:

- HF omits `v_proj` for `attention_k_eq_v && !is_sliding`, using key projection as value input.
- Deliverance currently always loads `v_proj` for non-shared layers in `Gemma4Model.loadTransformerBlockWeights(...)`, lines 156-157.
- For the checked fixture, `attention_k_eq_v=false`, so this is not the current E2B fixture issue.

### Shared KV

HF shared KV:

```python
first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx >= 0
prev_layers = config.layer_types[:first_kv_shared_layer_idx]
self.store_full_length_kv = not self.is_kv_shared_layer and layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])
```

During forward:

```python
if self.is_kv_shared_layer:
    key_states, value_states = shared_kv_states[self.layer_type]
else:
    ... compute K/V ...

if past_key_values is not None and not self.is_kv_shared_layer:
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
if self.store_full_length_kv:
    shared_kv_states[self.layer_type] = key_states, value_states
```

Deliverance equivalent:

- shared layer detection: `Gemma4Config.getSharedKvSourceLayer(...)`, lines 170-182
- store-source detection: `Gemma4Config.storesSharedKvState(...)`, lines 184-194
- shared state map lifecycle: `Gemma4Model.withSharedKeyValues(...)`, lines 401-410
- shared state put/get: `Gemma4Model.getSharedKeyValues(...)` and `putSharedKeyValues(...)`, lines 266-288
- attention consume/store: `Gemma4CausalSelfAttention.forward(...)`, lines 152-163 and 265-282

Assessment: conceptually similar. Existing diagnostics showed batch prefill vs token-by-token prefill matched, and decode vs cold replay matched for tested continuation, reducing suspicion that shared KV is catastrophically broken.

### PLE

HF PLE model-level flow:

```python
per_layer_inputs = self.embed_tokens_per_layer(input_ids).reshape(
    *input_ids.shape,
    self.config.num_hidden_layers,
    self.hidden_size_per_layer_input,
)

per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
per_layer_projection = per_layer_projection.reshape(
    *inputs_embeds.shape[:-1],
    self.config.num_hidden_layers,
    self.hidden_size_per_layer_input,
)
per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale
```

HF layer-level PLE:

```python
if self.hidden_size_per_layer_input:
    residual = hidden_states
    hidden_states = self.per_layer_input_gate(hidden_states)
    hidden_states = self.act_fn(hidden_states)
    hidden_states = hidden_states * per_layer_input
    hidden_states = self.per_layer_projection(hidden_states)
    hidden_states = self.post_per_layer_input_norm(hidden_states)
    hidden_states = residual + hidden_states

hidden_states *= self.layer_scalar
```

Deliverance equivalent:

- token identity: `Gemma4Model.computeTokenIdentityPerLayerInputs(...)`, lines 430-439
- context projection: `Gemma4Model.computeProjectedPerLayerInputs(...)`, lines 441-451
- combine: `Gemma4PleSupport.combinePerLayerInputs(...)`, lines 15-24
- split: `Gemma4PleSupport.splitPerLayerInputs(...)`, lines 26-43
- layer PLE: `Gemma4TransformerBlock.forward(...)`, lines 93-120

Assessment: the high-level flow matches. This remains a candidate for subtle scale/shape issues, but not an obvious mismatch from static comparison.

### Decoder Layer Order

HF layer order:

```python
residual = hidden_states
hidden_states = input_layernorm(hidden_states)
hidden_states, _ = self_attn(...)
hidden_states = post_attention_layernorm(hidden_states)
hidden_states = residual + hidden_states

residual = hidden_states
hidden_states = pre_feedforward_layernorm(hidden_states)
hidden_states = mlp(hidden_states)
hidden_states = post_feedforward_layernorm(hidden_states)
hidden_states = residual + hidden_states

PLE block if present
hidden_states *= layer_scalar
```

Deliverance equivalent:

- `Gemma4TransformerBlock.forward(...)`, lines 67-129

Assessment: order appears to match.

### Final Norm And Logits

HF:

```python
hidden_states = self.norm(hidden_states)
logits = self.lm_head(hidden_states[:, slice_indices, :])
if self.config.final_logit_softcapping is not None:
    logits = logits / self.config.final_logit_softcapping
    logits = torch.tanh(logits)
    logits = logits * self.config.final_logit_softcapping
```

Deliverance:

- output norm and head setup: `Gemma4Model.loadOutputWeights(...)`, lines 111-132
- final logit softcap: `AbstractGeneratorSampler`, lines 197-204 for the current sampler path

Assessment: implemented and likely not the current issue.

## Findings

### Finding 1: Gemma3n Features Do Not Belong In Gemma4

Status: fixed/cleaned from Gemma4 code after comparison.

HF Gemma4 text path has no AltUp, no Laurel, and no activation sparsity. Any such additions should stay out of `model/gemma4` unless actual Gemma4 checkpoint weights/config prove otherwise.

### Finding 2: Attention Score Scaling Matches HF

Status: no action.

HF Gemma4 text attention scaling is `1.0`; Deliverance uses unscaled dot product. That matches.

### Finding 3: Text Attention Softcap Should Be Inactive

Status: no action unless a config contains text `attn_logit_softcapping`.

The Gemma4 fixture has final logit softcap but no text attention softcap. Deliverance will not apply attention softcap unless `attnLogitSoftCapping` is non-null.

### Finding 4: RoPE Is The Highest-Value Remaining Static Mismatch To Verify

Status: needs deeper check.

HF delegates proportional RoPE to `ROPE_INIT_FUNCTIONS["proportional"]`, passing `head_dim_key="global_head_dim"` for full attention. Deliverance has a hand-written proportional RoPE implementation in `Gemma4Config.precomputeRopeFreqs(...)`, lines 127-163.

This is the next best target because a small RoPE frequency/scaling mismatch can produce coherent early text and then semantic drift, while still passing batch-vs-token and decode-vs-cold replay checks.

### Finding 5: `attention_k_eq_v` Alternative Attention Is Not Fully Implemented

Status: likely not E2B fixture issue because fixture has `attention_k_eq_v=false`.

If a Gemma4 checkpoint sets `attention_k_eq_v=true`, HF omits `v_proj` for full attention and derives values from K projection before K norm/RoPE. Deliverance currently loads/uses `v_proj` for all non-shared layers. This should be fixed before supporting such checkpoint variants.

### Finding 6: Masking Is Equivalent Only For Plain Text, No Padding, Causal Use

Status: acceptable for current tests; not complete.

HF builds full and sliding causal masks and passes them to attention. Deliverance manually limits visible rows by position/window. This is equivalent for unpadded text-only causal generation, but it does not cover padding, bidirectional modes, image/audio token masks, or multimodal block sequence masks.

## Recommended Next Step

Do a focused RoPE parity check next:

1. Extract HF proportional RoPE formula from `ROPE_INIT_FUNCTIONS["proportional"]`.
2. Compare Deliverance `Gemma4Config.precomputeRopeFreqs(...)` against HF for:
   - `sliding_attention`
   - `full_attention`
   - positions 0, 1, 2, 128, 511, 512, 4096
   - first rotary pair and tail pairs
3. If different, patch RoPE before touching attention/KV again.

Do not reintroduce Gemma3n AltUp/Laurel/activation sparsity into Gemma4.

## Execution-Order Mapping

This table maps the HF Gemma4 text path to Deliverance in execution order. It is intended for line-by-line print parity work.

| Order | HF Operation | Deliverance Mapping | Status | Suggested Probe |
|---:|---|---|---|---|
| 1 | `Gemma4ForCausalLM.forward` calls `self.model(...)` with ids, masks, cache, `per_layer_inputs` | `Gemma4Model.batchForward(...)` embeds, computes PLE, calls `forwardGemma4(...)`; single-token path in `forward(...)` | approx | Print input ids, position ids/startPos, last hidden first8 |
| 2 | `hidden_states = outputs.last_hidden_state`; `logits = lm_head(...)`; optional final softcap | `Gemma4Model.loadOutputWeights()` plus sampler output norm/projection and final softcap | approx | Print hidden before final norm, after norm, logits pre/post softcap |
| 3 | Return `Gemma4CausalLMOutputWithPast` | Deliverance returns generation response / sampler token | different | Compare logits and next token, not object shape |
| 4 | Text forward requires exactly one of `input_ids` or `inputs_embeds` | Java has token-id paths and an embedding overload | approx | Print which Java entrypoint is used |
| 5 | `inputs_embeds = embed_tokens(input_ids)` scaled by `sqrt(hidden_size)` | `Gemma4Model.loadInputWeights`, lines 83-108 | match | Print raw/scaled embedding first8 and scale |
| 6 | If PLE enabled, call `get_per_layer_inputs(input_ids, inputs_embeds)` | `computePerLayerInputs(...)`, lines 373-390 | match for token-id path | Print token identity packed row |
| 7 | HF can recover token ids from `inputs_embeds` for PLE | Java embedding overload passes `perLayerInputs = null` | missing | If embedding overload is used, PLE is disabled |
| 8 | `embed_tokens_per_layer(input_ids).reshape(batch, seq, layers, ple_dim)` | `computeTokenIdentityPerLayerInputs(...)` and `splitPerLayerInputs(...)` | match | Print packed/split PLE slices |
| 9 | `per_layer_model_projection(inputs_embeds) * hidden_size^-0.5` | `computeProjectedPerLayerInputs(...)`, lines 445-448 | match | Print projection before/after scale |
| 10 | Reshape projection to `[batch, seq, layers, ple_dim]` | Java keeps packed then grouped-normalizes and splits | approx | Verify split equals packed slice for layer i |
| 11 | `per_layer_projection_norm(per_layer_projection)` | `Gemma4RmsNormSupport.applyInPlace(...)` | match | Print invRMS/weights/output for one layer slice |
| 12 | `(projection + token_identity) * 2^-0.5` | `Gemma4PleSupport.combinePerLayerInputs(...)` | match | Print identity/projection/combined first8 |
| 13 | HF initializes/uses DynamicCache | Java uses `KvBufferCache.KvBuffer` | approx | Print position/layer/KV row before and after write |
| 14 | HF `position_ids = arange + past_seen_tokens` | Java uses `startPos + batchIndex` | match | Print position ids/start position |
| 15 | HF builds full and sliding masks | Java uses `windowStart` and `visibleLength` procedural masking | approx | Print layer type, position, windowStart, visibleLength |
| 16 | HF precomputes RoPE embeddings per layer type before loop | Java precomputes `ropeFreqsByLayerType` in config and applies in attention | approx | Print cos/sin for positions 0,1,512,4096 |
| 17 | HF uses `shared_kv_states = UserDict()` | Java `withSharedKeyValues(...)` map lifecycle | match | Print shared map keys before/after store/read |
| 18 | HF layer loop passes per-layer PLE slice and layer-specific mask/RoPE | `forwardGemma4(...)`, lines 290-315 | match | Print layer input/output first8 |
| 19 | HF final text norm happens before returning model hidden states | Java final norm happens in sampler/output path | different boundary; generation-equivalent | Hidden comparisons must include final norm |
| 20 | Decoder: residual, input RMSNorm | `Gemma4TransformerBlock.forward`, lines 67-68 | match | Print `pre_attn_norm` |
| 21 | Decoder: self attention call | `attention.forward(...)`, lines 70-72 | match | Print q/k/v and attention stages |
| 22 | Decoder: post-attn norm then residual add | `maybeApplyNorm` then `accumulate`, lines 73-78 | match | Print post-attention residual |
| 23 | Decoder: pre-FF norm then MLP | lines 80-84 | match | Print MLP gate/up/product/down |
| 24 | HF optional MoE block | Deliverance Gemma4 does not implement MoE | missing if `enable_moe_block=true` | Print config `enableMoeBlock`; E2B fixture false |
| 25 | Decoder: post-FF norm then residual add | lines 86-91 | match | Print post-FF residual |
| 26 | Decoder: PLE gate/activation/multiply/project/norm/residual add | lines 93-115 | match | Print PLE gate, product, projection, norm |
| 27 | Decoder: `hidden_states *= layer_scalar` | lines 118-120 | match | Print layer scalar and hidden before/after |
| 28 | Attention: Q projection, Q norm, RoPE | `Gemma4CausalSelfAttention.forward`, lines 113-118, 135-138 | match | Print q after projection/norm/RoPE |
| 29 | Attention: K/V projection; V may reuse K when `attention_k_eq_v` | lines 120-129 | match for fixture where `attention_k_eq_v=false`; partial otherwise | Print config and V path |
| 30 | Shared KV layers read `shared_kv_states[layer_type]` | lines 152-163, 187-190 | match | Print shared K/V shape and first8 |
| 31 | Non-shared KV updates cache | lines 172-181 | approx | Print K/V row before cache and cached row |
| 32 | Store full-length shared KV | lines 265-282 | match | Print stored full K/V shape and first8 |
| 33 | Attention kernel: repeat KV, qk matmul, scaling, mask, fp32 softmax, value matmul | Java loops per position/head using dot product, softmax, SAXPY, lines 205-255 | approx | Compare one head’s attention vector |
| 34 | Attention output projection | lines 289-302 | match | Print value output and o_proj result |
| 35 | MLP: `down(act(gate(x)) * up(x))` | `VariableMLPBlock.forward(...)` | match | Print gate/up/product/down |
| 36 | RMSNorm uses fp32 mean square and optional scale | `RmsNorm` and `Gemma4RmsNormSupport` | approx | Print invRMS, dtype, weight, output |
| 37 | RoPE setup uses `ROPE_INIT_FUNCTIONS`, proportional with `head_dim_key=global_head_dim` for full | `Gemma4Config.precomputeRopeFreqs(...)` | approx | Compare inv_freq/cos/sin directly |
| 38 | RoPE apply `(x*cos) + rotate_half(x)*sin` | `applyRope(...)`, lines 370-386 | match | Print one pair before/after |

## Print-Probe Priority

If manual parity work is needed, use this order:

1. Prompt IDs and embeddings after scale.
2. PLE packed identity/projection/combined for layer 0 and layer 4.
3. RoPE cos/sin for `sliding_attention` and `full_attention` at positions 0, 1, 512.
4. Layer 0 hidden after input norm, attention output, post-attention residual, MLP output, final layer output.
5. First full-attention layer same probes.
6. Final hidden before final norm, after final norm, logits top-20.
