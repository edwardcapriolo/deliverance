# Gemma4 Text: Transformers vs Deliverance

This document is a structural comparison, not a claim of faithful numerical parity.
Transformers is always on the left; Deliverance is always on the right. Source line
numbers refer to the checked-out files in this workspace and should be refreshed when
either source moves.

## Scope And Sources

| Transformers (left) | Deliverance (right) |
|---|---|
| `/ai-code/transformers/src/transformers/models/gemma4/configuration_gemma4.py:87-240` | `/ai-code/del5/core/src/main/java/io/teknek/deliverance/model/gemma4/Gemma4Config.java:11-295` |
| `/ai-code/transformers/src/transformers/models/gemma4/modular_gemma4.py:1040-1122` | `/ai-code/del5/core/src/main/java/io/teknek/deliverance/generator/Gemma4CausalSelfAttention.java:21-345` |
| `/ai-code/transformers/src/transformers/models/gemma4/modular_gemma4.py:1040-1122` | `/ai-code/del5/core/src/main/java/io/teknek/deliverance/generator2/Gemma4CausalSelfAttention2.java:15-99` and `/ai-code/del5/core/src/main/java/io/teknek/deliverance/generator2/KvCacheSelfAttention2.java:117-487` |
| `/ai-code/transformers/src/transformers/models/gemma4/modular_gemma4.py:1170-1249` | `/ai-code/del5/core/src/main/java/io/teknek/deliverance/generator/Gemma4TransformerBlock.java:19-388` |
| `/ai-code/transformers/src/transformers/models/gemma4/modular_gemma4.py:1170-1249` | `/ai-code/del5/core/src/main/java/io/teknek/deliverance/generator2/TransformerBlock2.java:18-265` |
| `/ai-code/transformers/src/transformers/models/gemma4/modular_gemma4.py:1311-1530` | `/ai-code/del5/core/src/main/java/io/teknek/deliverance/model/gemma4/Gemma4Model.java:39-966` |
| `/ai-code/transformers/src/transformers/models/gemma4/modular_gemma4.py:1540-1607` | `/ai-code/del5/core/src/main/java/io/teknek/deliverance/model/gemma4/Gemma4Model.java:309-424` and generation/output callers in `:141-225,536-591` |
| `/ai-code/transformers/src/transformers/models/gemma4/modular_gemma4.py:1125-1167,1267-1304` | `/ai-code/del5/core/src/main/java/io/teknek/deliverance/model/gemma4/Gemma4MoeFeedForward2.java:25-157` and legacy MoE in `Gemma4TransformerBlock.java:197-343` |
| `/ai-code/transformers/src/transformers/models/gemma4/modular_gemma4.py:1356-1435` | `/ai-code/del5/core/src/main/java/io/teknek/deliverance/model/gemma4/Gemma4Model.java:722-923` and `Gemma4PleSupport.java:11-50` |
| `/ai-code/transformers/src/transformers/cache_utils.py:1871-2003` | `/ai-code/del5/core/src/main/java/io/teknek/deliverance/tensor/kv/` plus shared-KV code in `Gemma4Model.java:204-225,602-624` |

The generated `modeling_gemma4.py` is not treated as an independent conceptual source;
Transformers documents `modular_gemma4.py` as the source of the generated model.

## Naming And Shape Alignment

| Transformers name | Deliverance name | Structural result |
|---|---|---|
| `hidden_states` | `hiddenStates` | Used in the V2 block and documentation checkpoints |
| `residual` | `residual` | Same residual variable at attention and FF boundaries |
| `post_attention_layernorm` | `postAttentionNorm` | Same post-attention normalization boundary |
| `post_feedforward_layernorm` | `postFFNorm` | Same post-FF normalization boundary |
| `per_layer_input` | `perLayerInput` | Same layer-selected PLE slice |
| `shared_kv_states` | `sharedKeyValues` / `v2SharedKeyValues` | Same type/source concept, different ownership API |
| `apply_layer_scalar` (inline HF multiply) | `applyLayerScalar` | Explicit final block operation |
| HF MoE branch | `applyMoeBranch` | Explicit branch operation in both blocks |
| HF PLE branch | `applyPle` | Explicit branch operation in both blocks |
| HF expert dispatch | `applyMoeBranch` / `applyMoeExpert` | Explicit router and expert operations in V2 |

## Decoder Layer

| Transformers, `/ai-code/transformers/src/transformers/models/gemma4/modular_gemma4.py:1203-1249` | Deliverance, `/ai-code/del5/core/src/main/java/io/teknek/deliverance/generator2/TransformerBlock2.java:126-222` |
|---|---|
|```python\nresidual = hidden_states\nhidden_states = self.input_layernorm(hidden_states)\nhidden_states, _ = self.self_attn(...)\nhidden_states = self.post_attention_layernorm(hidden_states)\nhidden_states = residual + hidden_states\n\nresidual = hidden_states\nhidden_states = self.pre_feedforward_layernorm(hidden_states)\nhidden_states = self.mlp(hidden_states)\n```|```java\nTensorRef residual = embedding;\nTensorRef preAttentionHiddenStates = preAttentionNorm.map(ln -> ln.forward(embedding)).orElse(embedding);\npostAttention = attention.forward(preAttentionInput.tensor(), startPosition, kvSession, tensorReducer, phase, batchSize, sequenceLength, attentionMask);\nhiddenStates = maybeApplyNorm(postAttention, postAttentionNorm);\napplyResidual(hiddenStates, residual, "post_attention_residual");\nTensorRef preFeedForward = preFFNorm.map(ln -> ln.forward(hiddenStates)).orElse(hiddenStates);\npostFeedForward = ffBlock.forward(preFeedForwardInput.tensor(), tensorReducer, phase);\n```|
|```python\nif self.enable_moe_block:\n    hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)\n    hidden_states_flat = residual.reshape(-1, residual.shape[-1])\n    _, top_k_weights, top_k_index = self.router(hidden_states_flat)\n    hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)\n    hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)\n    hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)\n    hidden_states = hidden_states_1 + hidden_states_2\n\nhidden_states = self.post_feedforward_layernorm(hidden_states)\nhidden_states = residual + hidden_states\n```|```java\nTensorRef postFeedForwardNorm = moeBlock != null && moeDenseNorm.isPresent()\n        && moeInputNorm.isPresent() && moeOutputNorm.isPresent()\n        ? applyMoeBranch(postFeedForward, hiddenStates, tensorReducer, phase)\n        : maybeApplyNorm(postFeedForward, postFFNorm);\napplyResidual(postFeedForwardNorm, hiddenStates, "post_ff_residual");\n```|
|```python\nif self.hidden_size_per_layer_input:\n    residual = hidden_states\n    hidden_states = self.per_layer_input_gate(hidden_states)\n    hidden_states = self.act_fn(hidden_states)\n    hidden_states = hidden_states * per_layer_input\n    hidden_states = self.per_layer_projection(hidden_states)\n    hidden_states = self.post_per_layer_input_norm(hidden_states)\n    hidden_states = residual + hidden_states\nhidden_states *= self.layer_scalar\n```|```java\npostFeedForwardNorm = applyPle(postFeedForwardNorm, perLayerInput, batchSize);\npostFeedForwardNorm = applyLayerScalar(postFeedForwardNorm);\nreturn maybeApplyNorm(postFeedForwardNorm, preResponseNorm);\n```|

The legacy equivalent is `/ai-code/del5/core/src/main/java/io/teknek/deliverance/generator/Gemma4TransformerBlock.java:93-162`, with explicit `applyMoeBranch` at `:186-196` and `applyPle` at `:198-229`.

## Configuration And Model Forward

| Transformers, `/ai-code/transformers/src/transformers/models/gemma4/configuration_gemma4.py:185-224` | Deliverance, `/ai-code/del5/core/src/main/java/io/teknek/deliverance/model/gemma4/Gemma4Config.java:98-213` |
|---|---|
|```python\nif self.layer_types is None:\n    self.layer_types = [\n        "sliding_attention" if bool((i + 1) % 6) else "full_attention"\n        for i in range(self.num_hidden_layers)\n    ]\nif self.layer_types[-1] != "full_attention":\n    self.layer_types[-1] = "full_attention"\n```|```java\nif (config.get("layer_types") == null) {\n    ...\n    layerTypes.add(((i + 1) % 6) == 0 ? "full_attention" : "sliding_attention");\n}\nif (!layerTypes.isEmpty() && !Objects.equals(layerTypes.getLast(), "full_attention")) {\n    layerTypes.set(layerTypes.size() - 1, "full_attention");\n}\n```|
|```python\nself.per_layer_config = ...\n```|```java\ngetLayerHeadDim(layerType)\ngetLayerKeyValueProjectionHeads(layerType)\ngetLayerQueryProjectionLength(layerType)\ngetLayerKvLength(layerType)\n```|
|```python\nfirst_shared = num_hidden_layers - num_kv_shared_layers\n```|```java\ngetSharedKvSourceLayer(layerIndex)\nstoresSharedKvState(layerIndex)\n```|

HF text model flow is `embed_tokens -> PLE -> layers -> norm`; Deliverance's V2 flow is `Gemma4Model.batchForwardRef/forwardRef` (`Gemma4Model.java:152-201`) and the legacy flow is `batchForward/forward` (`:536-591`). PLE packing, projection, normalization, combination, and splitting are explicit at `Gemma4Model.java:722-846` and `:831-923`.

## Attention And KV

| Transformers, `/ai-code/transformers/src/transformers/models/gemma4/modular_gemma4.py:1070-1122` | Deliverance, `/ai-code/del5/core/src/main/java/io/teknek/deliverance/generator2/KvCacheSelfAttention2.java:131-159` and `Gemma4CausalSelfAttention2.java:56-94` |
|---|---|
|```python\nquery_states = self.q_proj(hidden_states).view(hidden_shape)\nquery_states = self.q_norm(query_states)\nquery_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)\n...\nif self.is_kv_shared_layer:\n    key_states, value_states = shared_kv_states[self.layer_type]\nelse:\n    key_states = self.k_proj(hidden_states).view(hidden_shape)\n    value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states\n    key_states = self.k_norm(key_states)\n    ...\n    value_states = self.v_norm(value_states)\n```|```java\nTensorRef[] qkv = projectQkv(input, phase);\nnormalizeQueryKey(qkv[0], qkv[1]);\nnormalizeValue(qkv[2]);\napplyRotaryEmbedding(qkv[0], qkv[1], startPosition);\nqkv = sharedKeyValues(qkv);\nif (writesCache(mode)) {\n    writeKvRows(kvSession, mode, qkv[1], qkv[2], startPosition);\n}\nTensorRef attended = attend(...);\n```|
|```python\nif past_key_values is not None and not self.is_kv_shared_layer:\n    key_states, value_states = past_key_values.update(...)\nif self.store_full_length_kv:\n    shared_kv_states[self.layer_type] = key_states, value_states\n```|```java\nprotected TensorRef[] sharedKeyValues(TensorRef[] qkv)\nprotected boolean writesCache(CacheExecutionMode mode)\nGemma4Model.storeSharedKeyValuesRef(...)\nGemma4Model.sharedKeyValuesRef(...)\n```|

The legacy attention performs the same conceptual stages in `Gemma4CausalSelfAttention.java:101-301`, with shared source storage at `:272-288`. Heterogeneous cache dimensions are configured by `Gemma4Model.configureGemma4KvWidths` (`:128-138`) and passed through the KV2 cache path.

## PLE

| Transformers, `/ai-code/transformers/src/transformers/models/gemma4/modular_gemma4.py:1356-1435` | Deliverance, `/ai-code/del5/core/src/main/java/io/teknek/deliverance/model/gemma4/Gemma4Model.java:831-923` |
|---|---|
|```python\nreturn self.embed_tokens_per_layer(input_ids).reshape(\n    *input_ids.shape, self.config.num_hidden_layers,\n    self.hidden_size_per_layer_input,\n)\n```|```java\ncomputeTokenIdentityPerLayerInputs(tokenIds, tokenIdentity, packedLength);\n...\nreturn splitPerLayerInputs(projected, batchSize, gemma4Config.numberOfLayers,\n        gemma4Config.hiddenSizePerLayerInput);\n```|
|```python\nper_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale\nper_layer_projection = self.per_layer_projection_norm(per_layer_projection)\nreturn (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale\n```|```java\ncomputeProjectedPerLayerInputs(embeddings, projected, packedLength, gemma4Config);\ncombinePerLayerInputs(projected, tokenIdentity, packedLength);\n```|
|```python\nhidden_states = self.per_layer_input_gate(hidden_states)\nhidden_states = self.act_fn(hidden_states)\nhidden_states = hidden_states * per_layer_input\n```|```java\napplyPle(hiddenStates, perLayerInput, batchSize)\n```|

The shared helper operations are `Gemma4PleSupport.combinePerLayerInputs` and `splitPerLayerInputs` (`Gemma4PleSupport.java:15-50`).

## MoE

| Transformers, `/ai-code/transformers/src/transformers/models/gemma4/modular_gemma4.py:1133-1167,1267-1304` | Deliverance, `/ai-code/del5/core/src/main/java/io/teknek/deliverance/model/gemma4/Gemma4MoeFeedForward2.java:49-157` |
|---|---|
|```python\nhidden_states = self.norm(hidden_states)\nhidden_states = hidden_states * self.scale * self.scalar_root_size\nrouter_probabilities = softmax(expert_scores, dim=-1, dtype=torch.float32)\ntop_k_weights, top_k_index = torch.topk(router_probabilities, k=...)\ntop_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)\ntop_k_weights = top_k_weights * self.per_expert_scale[top_k_index]\n```|```java\nops.groupedRmsNormInPlace(normalized, 1, config.embeddingLength, config.layerNormEps, routerScale2);\nops.multiplyInPlace(... config.embeddingLength^-0.5 ...);\n...\nfloat[] probabilities = computeRouterProbabilities(logits, row);\nint[] experts = selectTopKExperts(probabilities);\nfloat weight = probabilities[expert] / sum * scaleAt(perExpertScale, expert);\napplyMoeExpert(input, output, row, expert, weight, phase);\n```|

The legacy implementation is `Gemma4TransformerBlock.java:231-343`. The structure is aligned, but neither implementation is declared numerically equivalent here.

## Output And Generation

| Transformers, `/ai-code/transformers/src/transformers/models/gemma4/modular_gemma4.py:1575-1607` | Deliverance, `/ai-code/del5/core/src/main/java/io/teknek/deliverance/model/gemma4/Gemma4Model.java:309-328,402-423` |
|---|---|
|```python\nhidden_states = outputs.last_hidden_state\nhidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)\nlogits = self.lm_head(hidden_states)\nif self.config.final_logit_softcapping is not None:\n    logits = logits / self.config.final_logit_softcapping\n    logits = torch.tanh(logits)\n    logits = logits * self.config.final_logit_softcapping\n```|```java\nTensorRef norm = registerModelTensorRef(loadV2(resolveTextModelRoot() + "norm.weight", qType));\n...\nreturn new SampleOutputRef() {\n    public LayerNorm2 outputLayerNorm() { return new RmsNorm2(...); }\n    public TensorRef outputLogitsWeights() { return outputHead; }\n};\n```|

Deliverance generation enters `batchForwardRef`/`forwardRef` for V2 or the inherited sampler path for legacy execution. The output projection boundary is intentionally different in API shape from HF's returned logits object. Final pre-sampling logits must be compared before treating generation text as evidence.

## Per-Class Checklist

| Class / path | Implemented structural alignment | Missing or unproven |
|---|---|---|
| `Gemma4Config` | Defaults, layer types, per-layer dimensions, RoPE parameter grouping, shared-source helpers | Exact HF serialization/default behavior for every config variant; heterogeneous KV validation |
| `Gemma4Model` | Text root resolution, embedding scale, V1/V2 layer loading, PLE packing, output head selection, generation entry points | Persistent shared-KV history semantics; exact HF dtype boundaries; final logits parity |
| `Gemma4TransformerBlock` | Legacy residual order, explicit `applyMoeBranch`, `applyPle`, final scalar | MoE numerical parity; generic legacy hooks proven equivalent to HF fields |
| `TransformerBlock2` | HF-visible names/order, explicit `applyResidual`, `applyMoeBranch`, `applyPle`, `applyLayerScalar` | Real-checkpoint activation parity and ownership under all optional branches |
| `Gemma4CausalSelfAttention` | V1 Q/K/V norm, RoPE, mask/window, shared-KV storage/read, output projection | Persistent shared-KV history semantics; exact dtype boundaries; full bidirectional/multimodal masks |
| `Gemma4CausalSelfAttention2` | V2 per-layer dimensions, Q/K/V normalization hooks, RoPE, shared source selection | HF attention scale/dtype boundary and numerical attention parity; `attention_k_eq_v` V2 loading path |
| `KvCacheSelfAttention2` | Explicit projection, normalization, RoPE, shared-KV, cache write/read, attend, output projection stages | Heterogeneous KV cache validation and persistent shared-KV history semantics |
| `Gemma4MoeFeedForward2` | Provider-backed router projection, fp32-like probability loop, top-k/renormalization, packed expert projections | MoE/HF numerical parity, tie ordering, dtype boundaries |
| `Gemma4PleSupport` | Packed combine and layer splitting helpers | Exact HF dtype/reshape parity with a real tensor dump |
| Output/generation path | Final norm/head wiring and V1/V2 prefill/decode entry points | Final logits parity, tokenizer/template/BOS parity, sampler-boundary equivalence |
| Multimodal boundary | Text-only implementation deliberately kept separate from HF towers | Multimodal paths are not implemented/proven in Deliverance |

## Explicit Known Gaps

These are deliberately not hidden by this refactor:

- Persistent shared-KV history semantics across independent decode calls are unproven.
- Heterogeneous KV cache validation is unproven for every supported layer pattern and cache backend.
- Exact HF dtype boundaries, including embedding, RMSNorm, router, cache, and logits transitions, are unproven.
- MoE/HF numerical parity is unproven.
- Multimodal paths, including vision/audio features and block masks, are not proven.
- Final logits parity before sampling is unproven.

## Verification Performed / Required

The refactor must be checked with `git diff --check`, Java formatting checks used by the
repository, the core compile, and focused Gemma4 unit tests. Those checks validate source
integrity and local behavior; they do not establish faithful HF parity. A parity claim still
requires the same real checkpoint and token IDs, tensor checkpoints through every layer,
shared-KV source/consumer comparisons, final pre-sampling logits, and only then generation.
