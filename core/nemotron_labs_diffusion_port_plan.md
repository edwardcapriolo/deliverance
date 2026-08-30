# Nemotron Labs Diffusion Port Plan

For the user-facing support guide, generation examples, benchmark commands, and current runtime notes, see [Nemotron Labs Diffusion Support](nemotron_labs_diffusion.md).

## Goal

Port `nvidia/Nemotron-Labs-Diffusion-3B-Base` into Deliverance in small, source-backed slices with diffusion text generation as the primary goal. The point of this port is to measure whether diffusion decoding changes CPU text-generation throughput in a meaningful way. Autoregressive generation is an intermediate correctness and loader milestone only; another CPU AR decoder by itself is not a major deliverable.

The Base checkpoint is the first target because it has the same core architecture as the chat/instruct repo while avoiding the extra linear-spec LoRA payload. Use `nvidia/Nemotron-Labs-Diffusion-3B` later as a secondary validation target for chat/instruct behavior.

Upstream model facts verified from Hugging Face metadata, downloaded `config.json`, and custom code:

- Primary repository: `nvidia/Nemotron-Labs-Diffusion-3B-Base`
- Secondary repository: `nvidia/Nemotron-Labs-Diffusion-3B`
- Model type: `nemotron_labs_diffusion`
- Architecture: `NemotronLabsDiffusionModel`
- Base implementation: custom `Ministral3Model`
- Parameters: approximately 3.8B BF16 parameters
- Weights: single `model.safetensors`, approximately 7.66 GB
- Modes: autoregressive, diffusion block decoding, linear self-speculation
- Config highlights: `hidden_size=3072`, `intermediate_size=9216`, `num_hidden_layers=26`, `num_attention_heads=32`, `num_key_value_heads=8`, `head_dim=128`, `hidden_act=silu`, `vocab_size=131072`, `mask_token_id=100`, `block_size=32`, YaRN RoPE

Primary success criteria:

- Run diffusion text generation on CPU from the Base checkpoint.
- Report tokens/sec, function evaluations, accepted/transferred tokens, and profiler breakdowns for diffusion decoding.
- Compare diffusion CPU throughput against this same model in AR mode, not against unrelated model families.
- Preserve AR mode only as a baseline and verification tool for the shared weights/cache path.
- After the model can generate one or two crisp clean sentences, analyze and quantize it so later testing can run faster.
- Evaluate TurboQuant prefix cache safety only after baseline cache semantics and diffusion/AR behavior are understood.

## Download Status

Current local path:

- `/home/ecapriolo/.deliverance/nvidia_Nemotron-Labs-Diffusion-3B-Base`

Observed during implementation:

- Present: `.finished`
- Present: `README.md`
- Present: `chat_template.jinja`
- Present: `config.json`
- Present: `configuration_nemotron_labs_diffusion.py`
- Present: `generation_config.json`
- Present: `modeling_nemotron_labs_diffusion.py`
- Present: `modeling_ministral.py`
- Present: `special_tokens_map.json`
- Present: `tokenizer_config.json`
- Present: `tokenizer.json`
- Present: `model.safetensors`, size `7,663,347,000` bytes

Checkpoint tensor inventory has been run and confirms 237 BF16 tensors.

## Base Config Inventory

Downloaded `config.json` values for the first target:

- `architectures=["NemotronLabsDiffusionModel"]`
- `model_type="nemotron_labs_diffusion"`
- `auto_map.AutoConfig="configuration_nemotron_labs_diffusion.NemotronLabsDiffusionConfig"`
- `auto_map.AutoModel="modeling_nemotron_labs_diffusion.NemotronLabsDiffusionModel"`
- `torch_dtype="bfloat16"`
- `vocab_size=131072`
- `hidden_size=3072`
- `intermediate_size=9216`
- `num_hidden_layers=26`
- `num_attention_heads=32`
- `num_key_value_heads=8`
- `head_dim=128`
- `attention_bias=false`
- `mlp_bias=false`
- `attention_dropout=0.0`
- `hidden_act="silu"`
- `rms_norm_eps=1.0e-5`
- `max_position_embeddings=4096`
- `sliding_window=null`
- `bos_token_id=1`
- `eos_token_id=2`
- `tie_word_embeddings=false`
- `use_cache=false` in config and generation config, while upstream AR generation explicitly calls with `use_cache=true`
- `mask_token_id=100`
- `block_size=32`
- `dlm_paradigm="bidirectional"`
- `ar_loss_weight=1.0`
- `dlm_loss_weight=null`
- `dp_varying_mask_ratio=false`
- `attn_implementation="sdpa"`

RoPE parameters:

- `rope_type="yarn"`
- `rope_theta=1000000.0`
- `factor=0.25`
- `original_max_position_embeddings=16384`
- `beta_fast=32.0`
- `beta_slow=1.0`
- `mscale=1.0`
- `mscale_all_dim=1.0`
- `llama_4_scaling_beta=0.1`

Derived dimensions:

- Attention projection length: `num_attention_heads * head_dim = 4096`
- KV projection length: `num_key_value_heads * head_dim = 1024`
- GQA group size: `num_attention_heads / num_key_value_heads = 4`
- MLP ratio: `intermediate_size / hidden_size = 3`

## Upstream Runtime Behavior Inventory

Upstream `NemotronLabsDiffusionModel` behavior verified from `modeling_nemotron_labs_diffusion.py` and `modeling_ministral.py`:

- `NemotronLabsDiffusionModel` owns `encoder = Ministral3Model(diffusion_config)` and `diffusion_head = Linear(hidden_size, vocab_size, bias=false)`.
- `diffusion_config.diffusion_lm` defaults to `true` for `dlm_paradigm=bidirectional`.
- AR generation explicitly sets each layer attention `diffusion_lm=false` before generation.
- AR generation bypasses the wrapper `forward` diffusion path and calls `encoder(...)` directly.
- AR prefill creates `DynamicCache`, uses `cache_position = arange(prompt_len)`, uses matching `position_ids`, calls `encoder(... use_cache=true ...)`, then projects the final hidden state through `diffusion_head`.
- AR decode loops one token at a time, advances `cache_position` by absolute sequence position, updates cache, and projects only the last hidden state.
- Diffusion generation appends `block_length` mask tokens, denoises masked positions by confidence transfer, then runs a causal post-block forward to update KV cache and seed the next block.
- Linear self-speculation alternates bidirectional draft and causal verify while cropping cache to the accepted prefix.
- `Ministral3Model.forward` applies final `encoder.norm` after all decoder layers.
- Decoder layer order is `residual -> input_rmsnorm -> self_attention -> residual add -> post_attention_rmsnorm -> mlp -> residual add`.
- MLP formula is `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- Attention formula applies Q/K/V projections, RoPE to Q/K, Llama-4 query scaling, optional cache update/concat, GQA `repeat_kv`, attention softmax, then `o_proj`.
- In diffusion mode, attention is bidirectional by passing no causal mask and `is_causal=false`.
- In AR mode, causal masking is enabled by passing `use_causal_mask=true` or by the explicit `ar_generate` path with cache positions.

## Feature Inventory And Gaps

The first implementation must not treat this as plain Mistral. Required features and current implications:

- Custom config class is required for `model_type=nemotron_labs_diffusion`, diffusion fields, and YaRN fields.
- Weight names are expected to be under `encoder.*` plus `diffusion_head.weight`; confirm with the completed safetensors header before loader code.
- Existing `Config` precomputes RoPE from `rope_theta` and a simple `factor`; this is not enough for upstream YaRN semantics.
- Existing attention scaling supports a fixed `attentionMultiplier`; upstream applies position-dependent `_get_llama_4_attn_scale(cache_position, beta, original_max_position_embeddings)` to queries after RoPE.
- Existing Llama/Mistral reuse must be validated for norm order, residual placement, GQA mapping, RoPE layout, cache position handling, and output projection shape.
- Existing tokenizer fetch needed updates to include custom Python files, `generation_config.json`, and `special_tokens_map.json`; those changes are part of setup work.
- `tie_word_embeddings=false`, so `diffusion_head.weight` must be loaded separately and must not fall back to embeddings unless the tensor inventory proves a tied layout.
- `use_cache=false` in config should not disable runtime AR cache; upstream AR generation explicitly uses cache despite the config value.
- `sliding_window=null`, so first AR path needs standard causal attention only.
- Diffusion mode needs bidirectional attention and mask-token block denoising, but this is not required for the first AR milestone.
- Block-diffusion flex attention exists upstream for `dlm_paradigm=block_diff`, but Base config uses `bidirectional`; do not implement block-diff flex masks until the base bidirectional diffusion path is needed.

## Current Implementation Status

Implemented in the first pass:

- `NemotronLabsDiffusionConfig` parses the Base config and diffusion-specific fields.
- Model registry and `AutoModelForCausaLm` support `model_type=nemotron_labs_diffusion`.
- `ModelFetcher` downloads Nemotron custom-code and metadata files needed for source inspection: `*.py`, `generation_config.json`, and `special_tokens_map.json`.
- `NemotronLabsDiffusionModel` loads `encoder.*` tensors and `diffusion_head.weight`.
- AR baseline generation works through the existing local generation stack using the same model weights.
- Initial diffusion generation is the default `model.generate(...)` path and also works through `generateDiffusion(...)`, using full-sequence bidirectional denoising over prompt plus active block. The default CPU path currently uses one-token micro-blocks so callbacks emit regularly; larger true diffusion blocks can be tested by calling `generateDiffusion(..., blockLength, threshold)` directly.
- AR remains available through `generateArBaseline(...)` for same-model comparison.
- The diffusion path uses provider-backed projection, softmax, argmax, SAXPY, activation-multiply, and output projection. RoPE remains a local Java implementation because no provider primitive currently exists for this layout.
- Metrics are emitted under `nemotron_labs_diffusion.*` for load, embedding, layer/attention/MLP/logits projection, denoise steps, transferred tokens, mask tokens remaining, and NFE.

Current smoke results on this CPU-only environment:

- AR one-token smoke, prompt `The capital of France is`: about `0.099 tok/s`, `~10.1s` time-to-first-token.
- Diffusion one-token smoke, same prompt and one-token block: about `0.085 tok/s`, `nfe=1`, `~11.7s` total.
- Diffusion two-token smoke, prompt `Give me a history of Lake George`: about `0.062 tok/s`, `nfe=2`, `~32.3s` total.

Current limitations:

- Diffusion currently recomputes prompt plus generated prefix plus active block for each denoising step instead of using upstream's causal prefix cache plus bidirectional block attention.
- YaRN RoPE is not HF-parity yet; existing Deliverance RoPE precomputation is still simpler than upstream Transformers YaRN.
- The diffusion path has Java control loops around attention heads/positions and RoPE. Heavy vector math uses tensor operations, but this path should be profiled and gradually moved toward TensorPlan/provider kernels where practical.
- No clean-sentence quality gate has passed yet; do not quantize this model based only on current one-token smokes.
- TurboQuant prefix cache has not been validated for Nemotron and must remain off/experimental.

## Non-Functional Requirements

These are acceptance criteria for all new Nemotron implementation work.

1. Prefer tensor operations over hand-written loops.

   Heavy math must use `TensorOperations`, existing kernels, or TensorPlan-composed tensor operations. Java loops are acceptable for request/control flow, bounds checks, token arrays, or tiny metadata operations, but not for production tensor math.

2. Prefer TensorPlan steps over custom mappers.

   New tensor transformations should be expressed as TensorPlan operations when an existing operation can describe the work. Custom TensorPlan mappers are allowed only when no existing TensorPlan primitive or tensor operation can express the shape/layout transformation cleanly.

3. Keep execution traceable.

   Major model steps should be implemented through TensorPlan where practical so they can be traced and inspected. If a path cannot use TensorPlan yet, document why and keep the boundary narrow.

4. Add profiler instrumentation.

   New hot or semantically important paths must add `InferenceProfiler` timers/counters/meters. Minimum coverage should include embedding, Q/K/V projection, attention, output projection, MLP gate/up/down projection, RoPE, cache updates, logits projection, diffusion denoising steps, and self-speculation draft/verify phases when those exist.

5. Add clear tests.

   Every behavior change needs focused tests. Prefer unit tests for config parsing, masks, tensor layouts, RoPE math, and helper logic. Use integration tests where checkpoint loading or real model wiring is the actual risk.

6. Add clear documentation.

   Source-backed behavior, intentionally incomplete semantics, and unsupported modes must be documented in code comments or markdown. Documentation should distinguish smoke support from HF-parity support.

7. Mirror upstream Hugging Face tests where they exist.

   Before implementing a slice, search upstream custom code, model repository tests, and Transformers tests for matching behavior. Port test names and cases where practical, converting Python snake_case names to Java camelCase.

8. Include smaller-shape smoke tests.

   Large-model checks are useful, but each major path should also have a small synthetic or tiny-shape smoke test that runs quickly and validates shape/layout/control-flow assumptions without requiring the full 3B checkpoint.

9. Confirm behavior against upstream; do not assume existing Deliverance implementations are perfect.

   Existing Llama/Mistral-style code can be reused only after the relevant behavior is checked against Nemotron's upstream custom source and local tests. Mistral support in Deliverance has not been fully vetted, so matching a Mistral-shaped implementation is not sufficient proof of correctness. For each reused component, document the upstream method or formula being matched and add focused tests for the specific assumption.

## Source Artifacts To Inspect

Before implementation, save or inspect these upstream files:

- `config.json`
- `generation_config.json`
- `configuration_nemotron_labs_diffusion.py`
- `modeling_nemotron_labs_diffusion.py`
- `modeling_ministral.py`
- `tokenizer_config.json`
- `tokenizer.json`
- `chat_template.jinja`
- `model.safetensors` header/tensor inventory

## Checkpoint Layout Work

Use `SafetensorsInspector` or `DefaultWeightLoader.tensorInfoMap()` to verify exact names and shapes before writing loader code. Expected families from upstream code are likely:

- `encoder.embed_tokens.weight`
- `encoder.layers.N.self_attn.q_proj.weight`
- `encoder.layers.N.self_attn.k_proj.weight`
- `encoder.layers.N.self_attn.v_proj.weight`
- `encoder.layers.N.self_attn.o_proj.weight`
- `encoder.layers.N.mlp.gate_proj.weight`
- `encoder.layers.N.mlp.up_proj.weight`
- `encoder.layers.N.mlp.down_proj.weight`
- `encoder.layers.N.input_layernorm.weight`
- `encoder.layers.N.post_attention_layernorm.weight`
- `encoder.norm.weight`
- `diffusion_head.weight`

Do not assume these names are correct until the safetensors header confirms them.

## Implementation Slices

### Slice 1: Metadata And Config

- Add `NemotronLabsDiffusionConfig`.
- Register `model_type=nemotron_labs_diffusion` in model support.
- Parse AR and diffusion-specific fields, including `mask_token_id`, `block_size`, `dlm_paradigm`, and YaRN RoPE parameters.
- Add config parse tests using upstream `config.json` values.

Acceptance:

- Config test proves key fields match upstream.
- No generation behavior added yet.

### Slice 2: Checkpoint Inventory And Loader Skeleton

- Download or inspect the real checkpoint metadata.
- Add a checkpoint-load integration test, tagged with existing long-test conventions.
- Load representative weights and register model-lineage tensors.
- Document exact tensor names and shape formulas.

Acceptance:

- Full checkpoint opens locally.
- Representative tensors match config-derived shapes.
- Missing tensors fail with explicit messages.

### Slice 3: AR Weight Loading

- Reuse existing Llama/Mistral-style transformer blocks only where source-backed checks show they match Nemotron's custom `Ministral3Model` behavior.
- Use `diffusion_head.weight` as output logits weight.
- Confirm GQA dimensions: `num_attention_heads * head_dim` for Q and `num_key_value_heads * head_dim` for K/V.
- Keep tensor projections on provider-backed paths.
- Record any reused Llama/Mistral assumptions in this plan or a companion support document, including norm ordering, residual placement, RoPE permutation, cache position handling, attention scaling, and MLP activation/projection order.

Acceptance:

- Model initializes with all AR-required tensors.
- No diffusion mode implemented in this slice.
- Tests cover each reused Llama/Mistral assumption needed for AR initialization and forward execution.

### Slice 4: AR Smoke Generation Baseline

- Implement autoregressive forward/generation first as a baseline and correctness scaffold.
- Add a small-shape smoke test for forward path.
- Add a real-checkpoint one-token or two-token smoke test if disk/runtime allow.
- Add profiler instrumentation for AR prefill/decode phases.

Acceptance:

- `model.generate(...)` returns tokens from the 3B checkpoint.
- Tests assert non-empty token output and finite logits.
- Output quality is not claimed as HF parity until RoPE/logit parity is tested.
- This slice is not considered the main project outcome; it exists to support and compare diffusion mode.

### Slice 5: YaRN And Llama-4 Attention Scaling

- Verify upstream `modeling_ministral.py` RoPE implementation.
- Implement or adapt Deliverance RoPE support for `rope_type=yarn` and `_get_llama_4_attn_scale`.
- Add direct numeric tests against small upstream-derived examples where possible.

Acceptance:

- Short-position and scaled-position RoPE tests pass.
- AR logits are ready for HF comparison.

### Slice 6: HF Parity Checks

- Compare Java logits against HF for a tiny prompt if Python dependencies and hardware allow.
- If full 3B parity is too expensive, build a small synthetic checkpoint with upstream-compatible tensor names and shapes.

Acceptance:

- Document numeric tolerances and remaining gaps.
- Do not present smoke generation as parity if parity was not measured.

### Slice 7: Diffusion Block Decoding

- Implement mask-token block generation matching upstream `generate(...)`.
- Implement confidence-based transfer token selection.
- Use TensorPlan/tensor operations for logits, probability, argmax/confidence, and mask updates where practical.
- Add profiler counters for denoising steps, transferred tokens, remaining mask tokens, and NFE.
- Add CPU throughput reporting for diffusion mode, including tokens/sec and tokens per forward/function evaluation.
- Compare against the AR baseline from Slice 4 using the same prompt, max token count, tensor provider, dtype, and hardware.

Acceptance:

- Diffusion smoke generates tokens with `block_length=32` or smaller configured block size in tests.
- Small-shape tests cover transfer scheduling and threshold behavior.
- CPU diffusion benchmark output is sufficient to tell whether diffusion is materially different from AR on this machine.

### Slice 8: Linear Self-Speculation

- Implement draft/verify mode after AR and diffusion paths exist.
- Preserve shared KV cache semantics.
- Add counters for draft tokens, accepted tokens, rejected tokens, acceptance length, and NFE.

Acceptance:

- Self-speculation smoke returns tokens.
- Acceptance statistics are observable.

### Slice 9: Quality Gate Before Quantization

- Generate a small set of clean, human-readable AR and diffusion samples from the Base checkpoint.
- Use prompts that exercise ordinary sentence completion and short instruction following.
- Record prompt, settings, output text, token count, elapsed time, NFE, and tensor provider.
- Do not quantize based only on shape/load/smoke success; require at least one or two crisp clean generated sentences first.

Acceptance:

- We have concrete sample outputs demonstrating the model path is not just returning arbitrary tokens.
- Remaining semantic caveats are documented before quantization changes numerical behavior.

### Slice 10: Quantization For Faster Testing

- Inspect whether existing Quantize On Demand and JQ4 tooling can process `NemotronLabsDiffusionConfig` and `encoder.*` / `diffusion_head.*` tensor names.
- First Q.O.D. policy: quantize only `encoder.layers.*` attention and MLP projection weights.
- First Q.O.D. policy must keep dense: `encoder.embed_tokens.weight`, all RMSNorm weights, and `diffusion_head.weight`.
- Use target name `nvidia/Nemotron-Labs-Diffusion-3B-Base-JQ4` unless there is a concrete reason to choose a different local cache name.
- Add or update quantization metadata support only after the full-precision model produces plausible text.
- Prefer a quantized artifact for repeated local tests and benchmarks once parity/quality risk is understood.
- Benchmark BF16/F32 baseline versus quantized mode for AR and diffusion using the same prompts and generation settings.
- Do not test `withOutputHeadQuantization(DType.Q4)` until projection-only Q4 loads, runs, and has measured output drift.
- Possible later opt-in improvement: `withOutputHeadQuantization(DType.Q4)` reduced Nemotron Base QOD AR logits
  projection from about `7.5s` to about `0.65s` over a 25-token profile on ARM Mac, improving end-to-end AR from
  about `2.0 tok/s` to about `4.7 tok/s` while preserving plausible text. Keep this opt-in until output drift is
  characterized. Do not route this path through GPU decode without a targeted fix; enabling GPU decode for this profile
  crashed in Metal/AGX command-buffer code on ARM Mac.

Acceptance:

- Quantized checkpoint loads through the normal model path.
- AR and diffusion smoke tests pass on the quantized artifact.
- Benchmark output shows whether quantization materially improves local iteration speed.
- Manifest confirms projection weights are Q4 with `.qb` sidecars and embeddings/norms/`diffusion_head.weight` remain dense.
- Documentation states which tensors are quantized, which remain full precision, and any observed output drift.
- The clean-sentence quality gate result is recorded before using Q4 output for performance conclusions.

### Slice 11: TurboQuant Prefix Cache Safety

- Review `core/PrefixCache.md` and TurboQuant prefix-cache docs before changing behavior.
- First validate ordinary KV prefix-cache mechanics for Nemotron AR mode: block-aligned hits, suffix positions, cache row round trips, and decode start position.
- Then evaluate TurboQuant prefix cache separately; do not enable it by default until cache row numeric drift and generation behavior are understood.
- For diffusion, verify whether prefix cache applies only to the causal context/prompt portion, not mutable mask-token blocks that are re-denoised.

Acceptance:

- Prefix-cache mechanical tests pass for Nemotron AR mode.
- TurboQuant cache tests quantify row error and generation impact.
- The plan explicitly says whether TurboQuant is safe, unsafe, or still experimental for Nemotron diffusion.

## Testing Matrix

- Config parse tests from upstream JSON.
- Tensor layout tests for projections and GQA dimensions.
- RoPE/YaRN unit tests.
- Attention mask tests for AR, bidirectional, and block-diffusion masks.
- Small-shape forward smoke tests.
- Full checkpoint load test.
- Real 3B AR one-token smoke test.
- Diffusion block smoke test.
- CPU diffusion throughput smoke/benchmark using the Base checkpoint.
- Self-speculation smoke test.
- Clean-sentence quality gate before quantization.
- Quantized checkpoint load and generation smoke tests.
- Prefix-cache and TurboQuant prefix-cache safety tests.

## Observability Checklist

Add timers/counters under names prefixed with `nemotron_labs_diffusion.`. Candidate metrics:

- `nemotron_labs_diffusion.embedding`
- `nemotron_labs_diffusion.layer.forward`
- `nemotron_labs_diffusion.attention.qkv_projection`
- `nemotron_labs_diffusion.attention.rope`
- `nemotron_labs_diffusion.attention.softmax`
- `nemotron_labs_diffusion.attention.output_projection`
- `nemotron_labs_diffusion.mlp.gate_up_projection`
- `nemotron_labs_diffusion.mlp.activation_multiply`
- `nemotron_labs_diffusion.mlp.down_projection`
- `nemotron_labs_diffusion.cache.update`
- `nemotron_labs_diffusion.logits_projection`
- `nemotron_labs_diffusion.diffusion.denoise_step`
- `nemotron_labs_diffusion.diffusion.transferred_tokens`
- `nemotron_labs_diffusion.diffusion.mask_tokens_remaining`
- `nemotron_labs_diffusion.self_speculation.draft`
- `nemotron_labs_diffusion.self_speculation.verify`
- `nemotron_labs_diffusion.self_speculation.accepted_tokens`
- `nemotron_labs_diffusion.quantization.load`
- `nemotron_labs_diffusion.quantization.output_drift`
- `nemotron_labs_diffusion.prefix_cache.hit`
- `nemotron_labs_diffusion.prefix_cache.turboquant_error`

## Explicit Non-Goals For The First Slice

- No diffusion decoding in the first config/loader slice.
- No self-speculation before AR and diffusion work independently.
- No hidden scalar Java implementation for production tensor math.
- No claims of HF parity from shape-only or smoke-only tests.
- No celebration of AR-only support as the destination; AR is useful only insofar as it unlocks and benchmarks diffusion.
