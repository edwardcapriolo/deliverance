# DiffusionGemma Support Map

This document maps Hugging Face DiffusionGemma source and tests to a Deliverance port plan. It is a source-derived implementation guide, not a design sketch. The initial port should satisfy the small synthetic Hugging Face tests before any real 26B checkpoint work.

## Source Files

Local Hugging Face source:

- `/ai-code/transformers/src/transformers/models/diffusion_gemma/configuration_diffusion_gemma.py`
- `/ai-code/transformers/src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py`
- `/ai-code/transformers/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py`
- `/ai-code/transformers/src/transformers/models/diffusion_gemma/modular_diffusion_gemma.py`

Local Hugging Face tests:

- `/ai-code/transformers/tests/models/diffusion_gemma/test_modeling_diffusion_gemma.py`
- `/ai-code/transformers/tests/models/diffusion_gemma/test_generation_diffusion_gemma.py`

## Model Components

- `DiffusionGemmaConfig`: top-level multimodal block-diffusion config.
- `DiffusionGemmaTextConfig`: text encoder/decoder config.
- `DiffusionGemmaForBlockDiffusion`: generation model with `DiffusionGemmaModel` plus LM head.
- `DiffusionGemmaModel`: owns encoder and decoder.
- `DiffusionGemmaEncoderModel`: text/vision encoder; writes KV cache.
- `DiffusionGemmaEncoderTextModel`: text encoder stack.
- `DiffusionGemmaDecoderModel`: canvas decoder; reads encoder KV cache and does not update it.
- `DiffusionGemmaMultimodalEmbedder`: maps vision features into text embedding space.
- `DiffusionGemmaSelfConditioning`: combines canvas embeddings with previous-step logits-derived soft embeddings.
- `DiffusionGemmaGenerationMixin`: block-autoregressive diffusion generation loop.
- `EntropyBoundSampler`: accepts low-entropy denoiser tokens and renoises unaccepted positions.
- `StableAndConfidentStoppingCriteria`: stops denoising when accepted canvases are stable and logits are confident.
- `LinearTemperatureScheduleLogitsProcessor`: applies per-denoising-step temperature.

## Config Fields To Port

### `DiffusionGemmaTextConfig`

- `vocab_size`
- `hidden_size`
- `intermediate_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `num_global_key_value_heads`
- `global_head_dim`
- `hidden_activation`
- `max_position_embeddings`
- `initializer_range`
- `rms_norm_eps`
- `pad_token_id`
- `eos_token_id`
- `bos_token_id`
- `tie_word_embeddings`
- `attention_bias`
- `attention_dropout`
- `sliding_window`
- `layer_types`
- `final_logit_softcapping`
- `use_bidirectional_attention`
- `num_experts`
- `top_k_experts`
- `moe_intermediate_size`
- `rope_parameters`

Post-init behavior:

- If `use_bidirectional_attention == "all"`, set causal behavior off and rewrite `sliding_window = sliding_window / 2 + 1`.
- If `layer_types` is absent, use a six-layer pattern: five `sliding_attention` layers followed by one `full_attention` layer.
- Force the final layer to `full_attention`.
- Default RoPE parameters:
  - `sliding_attention`: `rope_type=default`, `rope_theta=10000.0`.
  - `full_attention`: `rope_type=proportional`, `partial_rotary_factor=0.25`, `rope_theta=1000000.0`.

### `DiffusionGemmaConfig`

- `text_config`
- `vision_config`
- `boi_token_id`
- `eoi_token_id`
- `image_token_id`
- `initializer_range`
- `tie_word_embeddings`
- `canvas_length`

Post-init behavior:

- Missing `text_config` becomes a default `DiffusionGemmaTextConfig`.
- Dict `text_config` becomes `DiffusionGemmaTextConfig`.
- Dict `vision_config` defaults `model_type` to `gemma4_vision` and resolves through HF `CONFIG_MAPPING`.

### Generation Config Defaults

HF `DiffusionGemmaGenerationConfig._get_default_generation_params()` returns:

```text
max_new_tokens = 256
max_denoising_steps = 48
sampler_config = EntropyBoundSamplerConfig(entropy_bound=0.1)
t_min = 0.4
t_max = 0.8
stability_threshold = 1
confidence_threshold = 0.005
```

Generation config accepts a limited AR-compatible subset: `max_length`, `max_new_tokens`, cache fields, and special token IDs. It rejects common AR sampling fields such as `do_sample`, `num_beams`, `temperature`, `top_k`, `top_p`, and repetition penalties.

## Tensor And Module Key Map

Use real HF module paths in tiny checkpoints where possible. Representative paths:

- Encoder text root: `model.encoder.language_model.*`
- Decoder text root: `model.decoder.*`
- Vision root: `model.encoder.vision_tower.*`
- Multimodal projector: `model.encoder.embed_vision.*`
- LM head: `lm_head.weight`
- Token embeddings: `embed_tokens.weight`
- Text layers: `layers.<i>.*`
- Attention projections: `self_attn.q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `o_proj.weight`
- Attention norms: `self_attn.q_norm.weight`, `k_norm.weight`, `v_norm.weight` where present.
- Layer norms: `input_layernorm.weight`, `post_attention_layernorm.weight`, `pre_feedforward_layernorm.weight`, `post_feedforward_layernorm.weight`, plus `_1`, `_2`, and `pre_feedforward_layernorm_2` variants for MoE/extra feed-forward paths.
- MLP: `mlp.gate_proj.weight`, `mlp.up_proj.weight`, `mlp.down_proj.weight`
- Router: `router.norm.weight`, `router.proj.weight`, `router.scale`, `router.per_expert_scale`
- Experts: `experts.gate_up_proj`, `experts.down_proj`
- Self-conditioning: `decoder.self_conditioning.pre_norm.weight`, `post_norm.weight`, and any projection weights in that module.
- Vision embedder: `embed_vision.embedding_pre_projection_norm.weight`, `embedding_projection.weight`
- RoPE buffers are not model weights but are layer-type keyed: `full_attention_inv_freq`, `sliding_attention_inv_freq` and corresponding original values.

Tied/shared behavior:

- Encoder text trainable parameters are tied to decoder parameters except decoder self-conditioning weights.
- `lm_head.weight` ties to decoder token embeddings.
- Encoder and decoder token embeddings are tied.

Shape formulas for tiny checkpoints:

- Embedding: `[vocab_size, hidden_size]`.
- Q projection: `[local_attention_dim, hidden_size]`, where full layers may use global head dimension/head count.
- K/V projection: `[local_kv_dim, hidden_size]`.
- O projection: `[hidden_size, local_attention_dim]`.
- MLP gate/up: `[intermediate_size, hidden_size]`.
- MLP down: `[hidden_size, intermediate_size]`.
- Router projection: `[num_experts, hidden_size]`.
- Expert packed gate/up: shape must reflect two projections per expert. The exact layout must be confirmed from HF code before writing checkpoint data.
- Expert down: `[num_experts, hidden_size, moe_intermediate_size]` or packed equivalent, to be confirmed from HF `DiffusionGemmaTextExperts`.
- Vision projection: `[text_hidden_size, vision_hidden_size]` or HF equivalent, to be confirmed when porting vision tests.

## Forward Flow

### Encoder Text Model

```text
input_ids or inputs_embeds
-> scaled token embeddings
-> DynamicCache if no cache provided
-> position_ids from cache length
-> full/sliding attention masks
-> layer-type RoPE embeddings
-> encoder layers
-> final RMSNorm
-> BaseModelOutputWithPast(last_hidden_state, past_key_values)
```

Encoder attention writes/updates the KV cache.

### Multimodal Encoder

```text
input_ids / inputs_embeds
-> detect image placeholder tokens
-> replace image token IDs with text pad token for text embeddings
-> vision_tower(pixel_values, image_position_ids)
-> multimodal embedder projects vision features
-> masked_scatter image features into text embeddings
-> encoder text model
```

If image tokens appear without `pixel_values`, generation emits a warning in HF.

### Decoder Model

```text
decoder_input_ids (canvas)
-> scaled token embeddings
-> optional self_conditioning_logits softmax @ embedding weights
-> self-conditioning module
-> decoder position IDs after encoder cache length
-> bidirectional decoder attention mask
-> decoder layers reading encoder KV cache
-> final RMSNorm
-> BaseModelOutput(last_hidden_state)
```

Decoder reads the encoder KV cache and does not update it. Passing `use_cache` to the decoder raises an error in HF.

### Block Diffusion LM Head

```text
DiffusionGemmaModel
-> lm_head
-> cast logits to fp32
-> tanh softcap using final_logit_softcapping
-> DiffusionGemmaBlockDiffusionOutputWithPast
```

## Attention And Mask Semantics

- Encoder attention can be causal or bidirectional depending on `use_bidirectional_attention`.
- Decoder attention is non-causal over the canvas.
- Decoder keys/values concatenate encoder cache entries with current canvas KV.
- Canvas positions attend bidirectionally to all valid canvas positions.
- Canvas positions attend to valid encoder KV cache positions.
- Full and sliding layer masks are distinct.
- Dynamic cache with no padding can return `None` masks.
- Static/compileable cache requires an explicit decoder attention mask.
- Sliding mask behavior differs between dynamic and static cache because static cache includes unfilled positions.

## Generation Algorithm

HF generation is block-autoregressive diffusion.

```text
prepare generation config
prepare cache
prepare position IDs and attention masks
prepare sampler, logits processors, stopping criteria

for each output canvas:
  encode uncached input tokens and update encoder KV cache
  initialize or reuse current canvas
  prepare decoder bidirectional attention mask

  for cur_step from max_denoising_steps down to 1:
    decoder_forward(current_canvas, cache, self_conditioning_logits)
    processed_logits = logits_processors(raw_logits, cur_step)
    denoiser_canvas = multinomial(softmax(processed_logits))
    argmax_canvas = argmax(processed_logits)
    accepted_canvas = sampler.accept_canvas(current_canvas, denoiser_canvas, processed_logits)
    current_canvas = sampler.renoise_canvas(accepted_canvas)
    update diffusion stopping criteria
    self_conditioning_logits = processed_logits cast to embedding dtype
    optionally stream draft
    break if all rows finished denoising

  append argmax_canvas to output sequence
  apply AR stopping criteria and pad finished rows
  optionally stream committed canvas
  prepare masks and position IDs for next canvas

return sequences, tokens_per_forward, past_key_values
```

Entropy-bound acceptance:

```text
entropy = Categorical(logits).entropy()
sort entropy ascending
cumulative_entropy = cumsum(sorted_entropy)
accept positions where cumulative_entropy - current_entropy <= entropy_bound
```

Renoising:

```text
new_canvas = accepted token where accepted else random token from [0, vocab_size)
```

Adaptive stopping:

- Stable if accepted argmax canvas is unchanged for `stability_threshold` steps.
- Confident if mean entropy is below `confidence_threshold`.

Tokens per forward:

```text
valid generated tokens excluding prompt and pad / decoder forward passes
```

## HF Test Inventory And Port Classification

### Generation Utility Tests

Port first. These require no model weights.

| HF test | Behavior | Deliverance target |
| --- | --- | --- |
| `test_generation_config_interface` | accepts AR-compatible config subset | `DiffusionGemmaGenerationParametersTest` |
| `test_bad_diffusion_generation_config_parameterization` | rejects AR-only generation fields | same |
| `test_save_load_generation_config` | config persistence and sampler config round trip | same or JSON config test |
| `test_eb_sampler_initialize_canvas` | random canvas init | `EntropyBoundSamplerTest` |
| `test_eb_sampler_accept_canvas` | entropy-bound low-entropy acceptance | `EntropyBoundSamplerTest` |
| `test_eb_sampler_renoise_canvas` | accepted tokens stay; others randomize | `EntropyBoundSamplerTest` |
| `test_linear_temperature_schedule` | scheduled temperature math | `LinearTemperatureScheduleTest` |
| `test_stable_and_confident_stopping_criteria_confidence` | confidence threshold | `DiffusionStoppingCriteriaTest` |
| `test_stable_and_confident_stopping_criteria_stability` | stability threshold | `DiffusionStoppingCriteriaTest` |
| `test_tokens_per_forward` | single row metric | `TokensPerForwardTest` |
| `test_tokens_per_forward_batched` | batched metric | `TokensPerForwardTest` |

### Config And Basic Model Tests

Port after config classes and tiny checkpoint writer.

| HF test | Behavior | Deliverance target |
| --- | --- | --- |
| `test_config` | common config properties | `DiffusionGemmaConfigTest` |
| `test_model` | base forward shape with/without masks/decoder IDs | `DiffusionGemmaModelShapeTest` |
| `test_tied_weights` | encoder/decoder and LM head tying | `DiffusionGemmaTiedWeightsTest` |
| `test_use_cache_raises_exception` | decoder rejects `use_cache` | `DiffusionGemmaDecoderTest` |

### Mask Tests

Port after mask builder.

| HF test | Behavior | Deliverance target |
| --- | --- | --- |
| `test_diffusion_decoder_mask_no_cache_raises_exception` | cache required | `DiffusionGemmaMaskTest` |
| `test_diffusion_decoder_mask_dynamic_cache` | dynamic no-padding shortcut | same |
| `test_diffusion_decoder_mask_dynamic_cache_left_padding` | left padding materialized masks | same |
| `test_diffusion_decoder_mask_dynamic_cache_beyond_sliding_window` | dynamic cache beyond sliding window | same |
| `test_diffusion_decoder_mask_static_cache` | static cache materialized mask | same |
| `test_diffusion_decoder_mask_static_cache_bad_attention_mask` | invalid static mask raises | same |
| `test_diffusion_decoder_mask_static_cache_beyond_sliding_window` | static + sliding + padding | same |

### Generation Loop Tests

Port after text forward and generation loop.

| HF test | Behavior | Deliverance target |
| --- | --- | --- |
| `test_generate` variants | dynamic/static cache, custom sampler/temp/stopping | `DiffusionGemmaGenerateTest` |
| `test_generate_text_only` | no image inputs | same |
| `test_generate_from_generation_config` | generation config object | same |
| `test_generate_kwarg_overrides` | kwargs override config | same |
| `test_generate_with_past_key_values` | multi-turn cache reuse | same |
| `test_generate_beyond_sliding_window` | long generation and cache/mask behavior | `DiffusionGemmaLongGenerateIT` |
| `test_diffusion_streaming` | streamer draft/final behavior | defer; HF skips this |

### Vision And Integration Tests

Port after vision path.

| HF test | Behavior | Deliverance target |
| --- | --- | --- |
| `test_diffusion_gemma_chat_template` | text chat template | tokenizer/template test |
| `test_diffusion_gemma_chat_template_image` | image placeholder template | tokenizer/template image test |
| `test_diffusion_gemma_chat_template_with_thinking` | thinking template | tokenizer/template thinking test |
| `test_diffusion_gemma_forward_text_only` | real/full text numeric logits | defer until real checkpoint metadata gate |
| `test_diffusion_gemma_forward_with_image` | real/full image numeric logits | defer |
| `test_diffusion_gemma_forward_batched` | real/full batched numeric logits | defer |
| `test_diffusion_gemma_generate_with_image_batched` | real/full image generation | defer |
| `test_diffusion_gemma_generate_with_image_batched_long` | real/full long image generation | defer |
| minified forward/generate tests | smaller real HF checkpoints | port after synthetic path if checkpoint is accessible |

Skipped HF tests should remain deferred unless their TODO reason is resolved upstream: attention outputs, hidden states, missing weights init, TP plan, CPU/disk offload, and diffusion streaming.

## Tiny Config Proposal

Use a shape based on `DiffusionGemmaVisionText2TextModelTester`:

```text
batch_size = 3
vocab_size = 99 or padded 128/256 for provider friendliness
hidden_size = 32
num_attention_heads = 2
num_key_value_heads = 2
intermediate_size = 32
num_hidden_layers = 2
layer_types = [sliding_attention, full_attention]
num_global_key_value_heads = 2
global_head_dim = 16
moe_intermediate_size = 8
num_experts = 4
top_k_experts = 2
canvas_length = 16
seq_length = 25
image_token_id = 4
boi_token_id = 5
eoi_token_id = 6
mm_tokens_per_image = 2
vision_hidden_size = 16
vision_num_layers = 2
vision_num_heads = 4
vision_patch_size = 5
vision_image_size = 20
```

For Deliverance tests, keep the exact HF values where they affect mask counts and canvas shapes. Use padded vocab only in tests that need provider-friendly matmul dimensions.

## Risk List

- Generated HF files are derived from `modular_diffusion_gemma.py`; behavior should be read from generated files but design inheritance from modular source.
- Encoder/decoder weight tying is nontrivial.
- Decoder self-conditioning is unique to DiffusionGemma and must not be confused with normal token embeddings.
- Decoder reads KV cache but does not update it.
- Mask creation has static-vs-dynamic cache differences.
- Sliding attention mask has bespoke canvas handling.
- Full and sliding layer RoPE differ.
- Vision placeholder replacement can silently mismatch image token counts.
- MoE router/expert packed layouts need exact tensor shape verification before checkpoint writer work.
- Real model is large; synthetic tests must come first.

## Initial Implementation Order

1. Port config classes.
2. Port generation utility classes and non-model tests.
3. Port mask builder and mask tests.
4. Build tiny checkpoint writer and key/shape tests.
5. Implement text encoder/decoder forward shape tests.
6. Implement self-conditioning and tied-weight tests.
7. Implement block diffusion generation on tiny model.
8. Add vision path.
9. Gate real model metadata.
10. Benchmark and only then add SIMD/GPU-specific optimized canvas kernels.

## Non-Goals For Initial Work

- Do not download or run the full 26B model.
- Do not add speculative optimizations before parity tests exist.
- Do not use scalar Java loops as production kernels.
- Do not port vision before text and masks are stable.
