# DiffusionGemma Port Plan

## Goal

Port DiffusionGemma support into Deliverance by following Hugging Face source and tests first. The port should use small deterministic test models with real tensor names and scaled shape formulas before attempting real model loading.

The implementation must avoid slow scalar scaffolding in production paths. Panama is the Java baseline, native SIMD is required for CPU hot paths, and GPU shaders should be used only for large canvas operations where transfer costs can be amortized.

## Source References

Local Hugging Face files:

- `/ai-code/transformers/src/transformers/models/diffusion_gemma/configuration_diffusion_gemma.py`
- `/ai-code/transformers/src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py`
- `/ai-code/transformers/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py`
- `/ai-code/transformers/src/transformers/models/diffusion_gemma/modular_diffusion_gemma.py`
- `/ai-code/transformers/tests/models/diffusion_gemma/test_modeling_diffusion_gemma.py`
- `/ai-code/transformers/tests/models/diffusion_gemma/test_generation_diffusion_gemma.py`

## Phase 1: Source Mapping

Create `core/diffusion_gemma_support.md` before implementation.

Document:

- Model components: config, encoder, decoder, self-conditioning, multimodal embedder, generation mixin, sampler, stopping criteria.
- Config fields used by code and tests.
- Tensor/module key map with shape formulas and tied/shared weight notes.
- Encoder forward flow.
- Decoder forward flow.
- Attention variants: encoder, decoder, full, sliding, bidirectional canvas, vision behavior.
- Cache semantics: encoder updates cache; decoder reads cache and does not update it.
- Generation algorithm: outer canvas loop, inner denoising loop, entropy sampler, renoise, self-conditioning, stopping, canvas finalization.
- Full HF test inventory, with each test classified by implementation phase.
- Tiny config extracted from HF test model tester.
- Risk list.
- Recommended implementation order.

Acceptance criteria:

- Every HF DiffusionGemma test is named and classified.
- The doc contains enough tensor names and shape formulas to build a tiny checkpoint writer.
- The doc separates exact HF behavior from Deliverance implementation decisions.
- No model implementation begins before this document exists.

## Phase 2: Tiny Config And Checkpoint Writer

Build a synthetic DiffusionGemma checkpoint using real HF tensor names and scaled formulas.

Tiny target shape, based on HF tests:

- Small vocab, padded to provider-friendly size.
- Hidden size around `32`.
- `2` text layers.
- `2` attention heads.
- `2` KV heads.
- Canvas length around `16`.
- Intermediate size around `32`.
- `4` MoE experts.
- `2` top-k experts.
- Layer types: `sliding_attention`, `full_attention`.
- Small vision config only where tests require it.

Checkpoint writer must emit representative real keys for:

- Text embeddings.
- Encoder/decoder layer norms.
- Q/K/V/O projections.
- Q/K/V norms.
- MLP gate/up/down.
- Router.
- Experts gate/up/down.
- Self-conditioning layers.
- Vision tower/projector tensors required by tests.
- LM head / tied embeddings.

Tests:

- Assert representative tensor keys exist.
- Assert shape formulas.
- Assert loader can read the checkpoint.

## Phase 3: Configuration Port

Implement Deliverance config classes:

- `DiffusionGemmaConfig`
- `DiffusionGemmaTextConfig`
- vision config delegation or reuse
- generation config/value objects as needed

Port config behaviors:

- Canvas length.
- Image token IDs.
- Text and vision sub-configs.
- Layer types and last-layer full-attention correction.
- Full/sliding RoPE parameter maps.
- MoE fields.
- Bidirectional attention modes.

Tests:

- Config default construction.
- Config construction with text/vision configs.
- Layer type normalization.
- Canvas length propagation.
- Token ID propagation.

## Phase 4: Generation Utility Mechanics

Port the non-model generation tests first. These are deterministic and do not require weights.

Implement:

- `DiffusionGemmaGenerationParameters` or equivalent.
- `EntropyBoundSampler`.
- `LinearTemperatureSchedule`.
- `StableAndConfidentStoppingCriteria`.
- `tokensPerForward` calculation.

Tests from HF:

- Generation config interface.
- Bad generation parameter rejection.
- Save/load equivalent if applicable.
- Entropy-bound canvas acceptance.
- Renoise canvas.
- Linear temperature schedule.
- Stable/confident stopping criteria.
- Tokens-per-forward single and batched cases.

Implementation constraints:

- Production sampler should operate on batched logits tensors.
- Avoid large Java object loops in production paths.
- Small loops are acceptable only in tests/assertion helpers.

## Phase 5: Mask Builder

Implement DiffusionGemma mask construction:

- Encoder masks for full and sliding attention.
- Decoder bidirectional canvas masks.
- Static/dynamic cache mask behavior.
- Padding behavior.
- Canvas-to-cache attention.
- Canvas-to-canvas bidirectional attention.

Tests:

- Decoder attention mask without padding.
- Decoder mask with left padding.
- Decoder mask with static cache.
- Sliding attention mask shape and nonzero counts.
- Full attention mask shape and nonzero counts.

Implementation constraints:

- Use tensor fill/copy/vector operations for production masks.
- Avoid per-cell nested scalar construction for large masks.
- Exact small mask matrices may be asserted in tests.

## Phase 6: Text Encoder And Decoder Forward

Implement text-only forward before vision.

Components:

- Shared/tied embeddings.
- RMSNorm.
- Q/K/V/O projections.
- Q/K/V norm.
- Full/sliding RoPE.
- Encoder attention and cache writes.
- Decoder bidirectional attention over canvas plus encoder KV cache.
- Residual connections.
- MLP.
- MoE router and experts.
- Self-conditioning.
- Final norm and logits.

Tests:

- Text-only forward shape.
- Decoder canvas output shape.
- Logits shape.
- Tied weights.
- Encoder/decoder sharing.
- Decoder does not update cache.
- Self-conditioning changes output deterministically.
- Tiny deterministic forward parity against HF for selected cases.

Implementation constraints:

- No scalar production attention path.
- Panama provider is the minimum Java baseline.
- Native SIMD should cover CPU matrix hot paths.
- TensorPlan can represent fused subgraphs, but only enable rewrites after benchmarked wins.

## Phase 7: Vision Path

Port vision only after text path and masks are stable.

Components:

- Image placeholder detection.
- Vision tower support required by tests.
- Multimodal embedder.
- Image feature projection into text embedding space.
- Replacement of image placeholder embeddings.

Tests:

- Placeholder mask.
- Text plus image forward shape.
- Batched image forward shape.
- Image token count mismatch failure.
- Chat template image path if tokenizer assets permit it.

Implementation constraints:

- Image preprocessing can remain outside model forward.
- HF tests often pass tensors directly; start there.

## Phase 8: Block Diffusion Generation Loop

Implement the generation loop after model forward is testable.

Algorithm:

- Prepare generation config.
- Prepare cache.
- Outer loop over canvases.
- Encoder prefill or incremental prefill.
- Initialize current canvas.
- Reverse denoising loop.
- Decoder forward with self-conditioning logits.
- Logits processing.
- Sampling.
- Entropy-bound acceptance.
- Renoising.
- Adaptive stopping.
- Finalize canvas.
- Prepare next canvas.

Tests:

- Tiny text-only generate.
- Tiny generate with image where feasible.
- Fixed-seed deterministic tiny output against HF.
- Early stopping reduces forward count.
- Tokens-per-forward matches HF semantics.

Implementation constraints:

- This is not an autoregressive next-token loop.
- Decoder reads cache but does not write it.
- Encoder commits finalized canvases to the cache between blocks.

## Phase 9: Provider And GPU Work

Only optimize paths that are naturally large enough to amortize setup and transfer.

Candidate GPU work:

- Canvas self-attention dense QK/V.
- Batched output projection over canvas.
- MLP gate/up/down over canvas blocks.
- Router/expert grouped GEMM later.

Tests:

- GPU matches Panama within tolerance for tiny and medium shapes.
- GPU path is selected only when tensor sizes justify it.
- Fallback works when GPU is unavailable.

Benchmarks:

- Tiny canvas benchmark.
- Canvas lengths `16`, `32`, `64`, `256`.
- Compare Panama, SIMD, and GPU.

## Phase 10: Real Model Metadata Gate

Only after synthetic parity:

- Inspect real `google/diffusiongemma-26B-A4B-it` `config.json`.
- Inspect `model.safetensors.index.json`.
- Validate tensor keys and shape formulas.
- Do not download the full model unless explicitly requested.
- Evaluate quantized variants separately.

Known quantized references:

- `mlx-community/diffusiongemma-26B-A4B-it-4bit`
- `mlx-community/diffusiongemma-26B-A4B-it-5bit`
- `mlx-community/diffusiongemma-26B-A4B-it-6bit`
- `mlx-community/diffusiongemma-26B-A4B-it-8bit`
- `mlx-community/diffusiongemma-26B-A4B-it-mxfp4`
- `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`
- `mlx-community/diffusiongemma-26B-A4B-it-nvfp4`
- `RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic`
- `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4`

## Suggested Change Sets

### Change Set 1

- Add `core/diffusion_gemma_support.md`.
- Add config classes.
- Add generation utility classes.
- Port non-model generation utility tests.

### Change Set 2

- Add tiny checkpoint writer.
- Add mask builder.
- Port mask tests.

### Change Set 3

- Add text encoder/decoder forward.
- Port text-only shape/tied-weight/self-conditioning tests.

### Change Set 4

- Add block diffusion generation loop.
- Port tiny generation tests.

### Change Set 5

- Add vision path.
- Port image/multimodal tests.

### Change Set 6

- Add provider/GPU optimized canvas kernels.
- Benchmark before enabling any optimization by default.

## Non-Goals For Initial Port

- Full real model download.
- Production quality quantized DiffusionGemma loading.
- GPU-first execution without CPU parity.
- Ad hoc speculative optimizations without benchmark proof.
- Scalar Java production kernels used as final implementation.
