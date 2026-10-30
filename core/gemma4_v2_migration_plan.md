# Gemma4 V2 Migration Plan

## Goal

Move the Gemma4 text decoder from the legacy `AbstractTensor`/`KvBufferCache` path to
TensorRef/Lighter/KVCache2 while preserving the current Gemma4 architecture:

- hybrid sliding and full attention layers
- per-layer head dimensions and KV-head counts
- per-layer RoPE parameters
- shared KV layers
- per-layer input embeddings (PLE)
- dense and MoE feed-forward blocks
- bidirectional attention modes used by the text configuration

Multimodal audio, image, video, processor, and assistant paths remain out of scope for
the text-decoder migration.

## Phase 1: V2 Attention Infrastructure

- Allow `KvCacheSelfAttention2` subclasses to provide per-layer attention dimensions.
- Add a Gemma4 V2 attention class with:
  - Q/K RMS normalization
  - per-layer RoPE dimensions/frequencies
  - sliding/full/bidirectional masking
  - heterogeneous query/KV projection widths
- Add focused parity tests for projection shapes, RMS normalization, and RoPE.

## Phase 2: KV Sharing

- Add request-local TensorRef shared-KV state to `Gemma4Model`.
- Store and consume shared key/value rows through KVCache2-compatible leases.
- Verify shared and non-shared layer outputs against the current legacy path on tiny
  deterministic checkpoints.

## Phase 3: V2 Transformer Blocks

- Port Gemma4's decoder block ordering to `TransformerBlock2`:
  - input norm
  - attention and residual
  - pre/post feed-forward norms
  - optional PLE injection
  - layer scalar
- Add V2 PLE projection, gating, and normalization.
- Add dense MLP V2 support with Gemma4 hidden-size rules.

## Phase 4: V2 MoE

- Port router projection, router scaling, expert selection, packed expert weights,
  and top-k expert accumulation.
- Add deterministic dense/MoE parity tests and expert-weight sensitivity tests.

## Phase 5: Runtime Wiring

- Add the Lighter constructor to `Gemma4Model`.
- Enable V2 only after all Phase 1-4 paths are available.
- Route `GEMMA4_TEXT` through the Lighter builder.
- Keep the legacy path available as an explicit fallback during migration.

## Phase 6: Validation

- Run all existing `Gemma4HfTextModelPortedTest` cases.
- Run provider checks with Panama and native SIMD where available.
- Run quantized and BF16 tiny checkpoints.
- Run the short real-model generation smoke test.
- Finally run the 90-token `Gemma4PromptIT.chatWithThinking` test and inspect:
  - repeated-token runs
  - control-token leakage
  - blank or malformed output
  - cached decode versus cold replay behavior

## Non-Goals

- Do not silently fall back from Gemma4 V2 to V1 after V2 is enabled.
- Do not remove the legacy path until the real E2B text generation test is stable.
- Do not use output text alone as proof of parity; retain hidden-state, KV, and logits
  checks for deterministic tiny fixtures.
