# EmbeddingGemma 2 Faithful Port Plan

## Objective

Port Google's `google/embeddinggemma-2` into Deliverance while preserving the
Transformers behavior, tensor layouts, numerical results, provider execution
strategy, and performance characteristics wherever Deliverance supports an
equivalent path.

The port is a full multimodal port. Implementation may proceed in dependency
order, beginning with the text backbone, but image, audio, video, and
multimodal fusion remain required scope rather than optional follow-up work.

GPU scope is intentionally limited to the text path in this plan. Image,
audio, and video processing and encoder execution will initially use CPU-side
implementations and existing provider-neutral tensor paths. Adding WebGPU or
native-GPU kernels for those modalities is explicitly out of scope and must be
planned separately.

Primary references:

- `transformers/src/transformers/models/embedding_gemma2/modular_embedding_gemma2.py`
- `transformers/src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py`
- `transformers/src/transformers/models/embedding_gemma2/configuration_embedding_gemma2.py`
- `transformers/src/transformers/models/embedding_gemma2/processing_embedding_gemma2.py`
- `transformers/src/transformers/models/embedding_gemma2/video_processing_embedding_gemma2.py`
- `transformers/src/transformers/models/embedding_gemma2/convert_embedding_gemma2_weights.py`
- `transformers/tests/models/embedding_gemma2/`
- `google/embeddinggemma-2` checkpoint metadata and model card

The Transformers modular implementation is the architectural source of truth.
Generated Transformers files are useful references but must not be treated as
the source when the modular file differs.

## Existing Gemma Code Is Untrusted

The existing Deliverance Gemma 3 and Gemma 4 implementations, including any
PLE/PLC-related implementation, are not correctness references. Previous
Gemma work produced corrupted output after extended generation, so those paths
must be treated as potentially defective even when they appear structurally
similar to EmbeddingGemma 2.

Existing Gemma code may be inspected for repository conventions, tensor-provider
integration points, allocation patterns, or candidate reusable infrastructure.
It must not be copied as a trusted implementation. Any reused operation,
algorithm, kernel, tensor layout, normalization, attention path, or PLE/PLC
logic requires:

- Independent derivation from the Transformers reference and model contract.
- First-principles unit tests with known mathematical results.
- Tiny-checkpoint numerical parity tests.
- Boundary, masking, dtype, offset, and shape tests where applicable.
- Long-sequence or repeated-execution tests when state, cache, or numerical
  drift could corrupt later output.
- Provider parity tests before production-path adoption.

Passing an existing Gemma 3 or Gemma 4 test is not evidence that reused code is
correct for EmbeddingGemma 2. The port must establish correctness independently
before relying on any such code.

## Deliverance Gaps

| Area | Current Deliverance | Required Work |
|---|---|---|
| Model registration | No `embedding_gemma2` | Add config/model registration |
| Text model | BERT and causal Gemma paths only | New EmbeddingGemma 2 encoder |
| Attention | Causal or BERT-style uniform attention | Bidirectional sliding/full attention |
| Head dimensions | Generally uniform | Per-layer local/global dimensions |
| PLE | Missing | Implement PLE projection and gating |
| Q/K/V norms | Missing for this path | Implement exact RMSNorm behavior |
| Output | Usually hidden size | 512 hidden to 768 projection |
| Pooling | Generic and BERT-specific | Masked mean plus float32 normalization |
| Prompts | No EmbeddingGemma task prompt API | Query/document/task prompt support |
| MRL | Missing | Truncate and re-normalize at 128/256/512/768 |
| Tokenizer | Gemma foundation exists | Validate special tokens and exact token IDs |
| Image | Missing | Processor, vision tower, patch positions |
| Audio | Missing | Feature extraction, audio tower, masks |
| Video | Missing | Frame sampling, processor, video tower |
| Multimodal batching | Missing | Ragged/nested modality inputs |
| Reference parity | No EmbeddingGemma fixtures | Stable reference fixtures and tolerances |
| Performance parity | No EmbeddingGemma implementation | Text provider benchmarks; CPU modality baselines |

## Rules For A Faithful Port

- Map the Transformers old and new execution paths side by side before
  declaring an implementation complete.
- Treat existing Gemma 3/Gemma 4 and PLE/PLC code as untrusted candidate
  infrastructure, never as an implementation oracle.
- Use the real checkpoint tensor names and shape formulas. Tiny checkpoints
  must be scaled versions of the real layout, not invented layouts.
- Preserve operation order, provider dispatch, memory layout, offsets, strides,
  masks, and fallback behavior unless a deviation is explicitly documented.
- Prove numerical parity within documented tolerances for FP32 and BF16 where
  supported.
- Do not silently replace an optimized provider operation with a scalar or
  naive implementation.
- Document every unsupported permutation with a nearby TODO identifying the
  affected path and unverified parity/performance behavior.
- Every changed or added method requires Javadoc, including private methods,
  covering contracts, shape assumptions, provider behavior, and fallbacks.
- Every post-change review must independently inspect the reference and
  Deliverance paths, tests, provider dispatch, and performance evidence.

### Edge-Performance Constraint

EmbeddingGemma 2 is explicitly intended to run on phones and constrained
consumer hardware. A numerically correct implementation that Java-orchestrates
one provider call per token, row, head, or PLE layer is not acceptable as the
production path.

- Keep token, sequence, and PLE workloads batched in contiguous TensorRef
  shapes.
- Prefer one provider operation over a flattened workload to repeated calls over
  individual rows.
- Use views/reshapes where possible; do not copy solely to change logical
  grouping.
- Keep PLE projection as one projection over all tokens, followed by one
  flattened per-layer RMSNorm operation.
- Execute each PLE residual block once over all tokens for the current layer,
  not once per token.
- Treat scalar loops as reference/fallback implementations only and mark them
  clearly with the missing fused/provider operation.
- Add allocation counts, provider-call counts, and warmed-up timing checks to
  characterization tests before accepting a production path.
- Do not claim the port is faithful if it is numerically equivalent but
  materially worse on a 10-core/16-GB laptop-sized target.

### Test Porting Rule

Every upstream EmbeddingGemma 2 test module and test case must have a named
Java counterpart. Preserve the upstream test module and test-case names in the
Java naming or an explicit mapping so coverage can be audited mechanically.
The Java test should directly reproduce the Python test's behavior, inputs,
assertions, and error semantics whenever the Deliverance API can express the
same contract.

If a test cannot execute yet because the required Deliverance implementation,
media backend, provider, fixture, or dependency is not available, add the
counterpart anyway and mark it ignored/disabled with a concise reason and the
upstream test reference. An ignored test is a tracked port gap, not completed
coverage.

If a test is skipped for expediency rather than blocked by a missing
capability, leave a nearby comment explaining exactly what was skipped, why it
was deferred, what behavior remains unverified, and what must be done before
the port can be considered complete. Do not silently omit tests or collapse
multiple upstream tests into an untraceable broad smoke test.

Tests that rely on Python-only mechanics should retain the same observable
contract through a Java-specific implementation. The port may change the
fixture or harness, but it must not weaken the assertion without documenting
the reason and residual risk.

## Phase 0: Freeze The Reference Contract

Before production implementation:

1. Record the exact Transformers checkout revision and model revision used as
   the reference.
2. Capture the model configuration, tokenizer configuration, Sentence
   Transformers metadata, processor configuration, and safetensors header.
3. Enumerate all checkpoint tensor names, shapes, dtypes, and namespaces.
4. Map each Transformers module and test to its intended Deliverance class or
   test contract.
5. Identify behavior that is Python-framework-specific and needs a Java API
   decision rather than a literal translation.

The checkpoint header and metadata must be committed or otherwise made
reproducible for tests. Do not download or commit the full model as a test
fixture.

## Phase 1: Text Backbone

This is the first implementation phase, but not the final product scope.

### Configuration and registration

- Add `EmbeddingGemma2TextConfig` and the composite configuration structure
  needed to deserialize the real `config.json`.
- Register `embedding_gemma2` and `embedding_gemma2_text` with `ModelSupport`.
- Preserve nested text, vision, and audio configuration fields even while only
  the text path is initially executable.
- Represent:
  - 262,144 vocabulary size.
  - 512 hidden size.
  - 2,048 intermediate size.
  - 24 layers.
  - Local/global layer types.
  - Local and global head dimensions.
  - Local/global RoPE theta values.
  - Sliding-window radius.
  - 768 output dimension.
  - RMSNorm epsilon.

### Text model execution

Implement a dedicated encoder rather than adapting the causal Gemma execution
path in place:

- Scaled Gemma token embeddings using the reference BF16 rounding behavior.
- Q, K, and V projections without bias.
- Q and K RMSNorm.
- Unscaled V RMSNorm.
- Per-layer GQA/MQA repetition.
- Bidirectional full-attention masks.
- Bidirectional symmetric sliding-window masks.
- Local and global RoPE parameters.
- GELU gated MLP.
- Input, post-attention, pre-feedforward, and post-feedforward RMSNorms.
- Residual ordering exactly matching the reference.
- Projection-only PLE calculation from input embeddings.
- Per-layer PLE gating and projection.
- Final RMSNorm.
- Learned 512-to-768 embedding projection.
- Encoder execution without causal KV-cache assumptions.

### Text embedding contract

- Tokenize with the Gemma tokenizer path.
- Preserve attention masks and padding semantics.
- Mean-pool only valid token positions.
- Perform L2 normalization in float32.
- Return 768 dimensions before optional MRL truncation.
- Add task-aware prompt formatting from
  `config_sentence_transformers.json`.
- Support asymmetric query/document encoding.
- Add MRL truncation at 768, 512, 256, and 128 dimensions.
- Re-normalize after truncation.

### Embedding Request And Post-Processing Contract

Transformers' `EmbeddingGemma2Model.forward()` does not accept MRL or
Sentence Transformers output options. Those options belong to the higher-level
embedding API. Deliverance must keep model inputs separate from embedding
post-processing so the same contract works for text and later multimodal
inputs.

Introduce an embedding request/options abstraction with, at minimum:

- Task prompt name, such as `SearchQuery`, `Document`, or `CodeRetrieval`.
- Explicit prompt override where a named prompt is insufficient.
- Optional `truncateDimension`.
- Optional `normalizeEmbeddings` applied after truncation.

The existing `embed(input, PoolingType)` API should retain its current default
behavior and return the native 768-dimensional result. The new options path
should apply this sequence:

```text
input -> task prompt -> tokenizer -> transformer -> mean pool
      -> native 768-dimensional embedding -> truncate leading dimensions
      -> optional float32 L2 normalization -> return
```

Do not pass `truncateDimension` or `normalizeEmbeddings` into the transformer
forward implementation. They are output-processing options, matching
Sentence Transformers' `encode()` behavior.

### MRL Testability Requirements

The options abstraction must be usable by tests without requiring the complete
multimodal API. Add deterministic text fixtures and test the following
dimensions:

- 768: native output.
- 512: first 512 dimensions.
- 256: first 256 dimensions.
- 128: first 128 dimensions.

Tests must assert:

- Output shape matches the requested dimension.
- Truncated values equal the leading dimensions of the full output before
  re-normalization.
- Post-truncation normalization produces unit-length vectors.
- Query and document encodings use the same requested dimension.
- Batch and single-input truncation agree.
- Unsupported or invalid dimensions have explicit behavior and tests.
- Reference parity against Sentence Transformers/Transformers at least at 256
  dimensions.

The reference fixture must preserve both the native 768-dimensional vector and
the expected truncated/normalized vectors. Do not test only final similarity;
the leading-dimension and norm invariants must also be asserted.

## Phase 2: Text Tests

Port the relevant Transformers tests and add Deliverance-specific tests.

### Configuration and checkpoint tests

- Config JSON deserialization and round trip.
- Real model type and nested config assertions.
- Real tensor-name assertions.
- Real shape formulas and scaled tiny-checkpoint shapes.
- Dtype and safetensors loading checks.
- Representative input, output, norm, attention, MLP, PLE, and projection
  tensor checks.

### Tiny numerical tests

Using a tiny checkpoint with real tensor names:

- Token embedding scaling.
- RMSNorm exact behavior.
- Q/K/V normalization.
- RoPE frequency generation for local and global layers.
- GQA/MQA head expansion.
- Bidirectional attention visibility.
- Sliding-window boundaries and tails.
- MLP gate/up/down ordering and activation.
- Residual and normalization ordering.
- PLE projection, reshape, normalization, and gating.
- Final projection shape and values.
- Masked mean pooling.
- Float32 L2 normalization.

### Behavioral tests

- Deterministic repeated inference.
- Weight sensitivity.
- Padding invariance.
- Sequence lengths around sliding-window boundaries.
- Odd lengths and non-divisible batch sizes.
- Batch versus single-input equivalence.
- Empty and invalid input behavior.
- Query/document prompt formatting.
- MRL truncation and post-truncation normalization.

### Reference parity tests

- Compare Deliverance against Transformers on deterministic text cases.
- Compare intermediate outputs where practical, not only final embeddings.
- Use documented tolerances for FP32 and BF16.
- Preserve versioned expected fixtures so tests do not depend on an internal
  Hub bucket or an untracked local environment.

## Phase 3: Text Provider And Performance Validation

- Run the text model through the production Panama provider.
- Compare against the naive/reference provider on small shapes.
- Exercise native SIMD when available.
- Verify tails, offsets, strides, row chunks, and non-divisible dimensions.
- Benchmark warmed-up production-shaped text inputs.
- Compare operation granularity and allocation behavior with the reference
  execution strategy.
- Document any intentional difference before proceeding.

GPU validation in this phase applies only to text operations. It does not
require GPU image, audio, or video kernels.

## Phase 4: Shared Multimodal Input And Processor API

Define the Java-side contracts before implementing modality towers:

- Image input representation.
- Audio waveform representation.
- Video frame/stream representation.
- Structured interleaved content.
- Per-sample modality lists.
- Ragged batch representation.
- Processed pixel tensors.
- Processed audio features and masks.
- Video frame counts and metadata.
- Placeholder/token expansion results.

Implement processor behavior equivalent to the Transformers processor:

- Gemma special tokens:
  - `<|image|>`
  - `<|video|>`
  - `<|audio|>`
  - Begin/end modality tokens.
- Automatic placeholder insertion.
- Manual placeholder preservation.
- Placeholder count validation.
- Modality-only inputs.
- Interleaved ordering.
- Chat-template behavior.
- Mismatched batch validation.
- Empty-input validation.

## Phase 5: Vision Encoder And Image Processor

The initial vision implementation is CPU-side. GPU image kernels, WebGPU
shaders, and native-GPU vision providers are not part of this plan.

### Image processor

- Decode RGB images.
- Rescale pixel values exactly as the reference.
- Preserve aspect ratio.
- Resize to patch-compatible dimensions.
- Extract 16x16 patches.
- Apply the pooling-kernel/token-budget rules.
- Produce 2-D patch position IDs.
- Produce dynamic image soft-token counts.

### Vision encoder

Port the required Gemma 4 vision components:

- Patch embedding.
- Vision positional handling.
- Axial RoPE.
- 16-layer vision encoder.
- Hidden size 768.
- 12 attention and KV heads.
- Vision MLP and normalization.
- Vision pooling.
- Vision-to-text projection.

### Vision tests

- Processor shape and resize tests.
- Patch count and token-count parity.
- 2-D position-ID tests.
- Axial RoPE numerical tests.
- Tiny vision forward tests.
- Image output shape tests.
- Image-only and image-plus-text parity.
- Multiple image ordering tests.
- Ragged image batch tests.

## Phase 6: Audio Encoder And Audio Processor

The initial audio implementation is CPU-side. GPU audio kernels, WebGPU
shaders, and native-GPU audio providers are not part of this plan.

### Audio processor

- Decode supported audio formats.
- Convert to mono.
- Resample to 16 kHz.
- Compute the reference mel features:
  - 128 feature bins.
  - 512 FFT length.
  - 320 frame length.
  - 160 hop length.
  - Reference padding and pre-emphasis behavior.
- Produce feature masks.
- Compute dynamic audio soft-token counts.

### Audio encoder

Port the required Gemma 4 audio components:

- Subsampling convolution frontend.
- Relative positional encoding.
- Audio attention.
- Audio feed-forward layers.
- Causal/light convolution blocks.
- 12-layer audio encoder.
- Hidden size 1024.
- Output projection to the text hidden size.

### Audio tests

- Feature extraction golden tests.
- Audio mask tests.
- Subsampling length arithmetic.
- Dynamic placeholder count parity.
- Tiny audio forward tests.
- Audio-only and audio-plus-text parity.
- Multiple audio segment ordering.
- Ragged audio batch tests.

## Phase 7: Video Processor And Fusion

The initial video implementation is CPU-side. GPU video preprocessing, WebGPU
shaders, and native-GPU video providers are not part of this plan.

### Video processor

- Decode frames.
- Sample at 1 FPS by default.
- Enforce the 32-frame default maximum.
- Uniformly subsample overflow.
- Reuse image preprocessing for frames.
- Preserve per-video frame counts.
- Support flattened frame tensors with ragged per-video grouping.
- Implement timestamp behavior and missing-FPS errors.

### Composite fusion

- Replace image placeholders with projected vision features.
- Replace audio placeholders with projected audio features.
- Replace video placeholders with projected per-frame vision features.
- Verify feature count equals placeholder count.
- Preserve interleaved ordering.
- Apply the correct shared attention mask.
- Mean-pool and normalize the complete multimodal sequence.

### Video and fusion tests

- Frame sampling and FPS tests.
- Frame budget and truncation tests.
- Metadata/error tests.
- Video token-count tests.
- Single-video versus batched-video equivalence.
- Text/image/audio/video interleaving tests.
- Mixed modality batches.
- Image-only, audio-only, video-only, and combined inputs.
- Manual placeholder ordering and validation.

## Phase 8: Full-Model Reference Parity

Add stable fixtures for:

- Text.
- Image.
- Audio.
- Video.
- Text plus image.
- Text plus audio.
- Text plus video.
- Image plus audio.
- Mixed batches.
- Manual placeholders.
- Query/document prompts.
- MRL dimensions.

Reference fixtures should be generated from the pinned Transformers/model
revision and stored in a compact, versioned format. Tests must record the
reference revision and dtype.

## Phase 9: Provider And Performance Parity

- Validate the text path through Panama and native providers where supported.
- Compare text against a naive/reference implementation on small
  deterministic cases.
- Measure warmed-up production-shaped text workloads.
- Establish CPU-side correctness and baseline timings for image, audio, and
  video paths.
- Track memory use for text-only, text+image, text+audio, and full multimodal
  configurations on the CPU modality baseline.
- Verify selective loading of unused modality towers.
- Document unsupported hardware or media backends explicitly.
- Defer GPU image, audio, and video optimization to a separate plan.

## Phase 10: Review And Completion

The port is complete only when:

1. Text, vision, audio, video, processor, and fusion paths are mapped against
   the Transformers implementation.
2. Real tensor names and shape formulas are validated.
3. Numerical parity tests pass within documented tolerances.
4. All processor validation and batching semantics are tested.
5. MRL, prompts, pooling, and normalization match the reference.
6. Text provider behavior has no unexplained material regression, and CPU
   modality baselines are recorded.
7. Missing functionality is documented with nearby TODOs.
8. An independent subagent review reports no unresolved findings.
9. Focused module tests and the appropriate compile checks pass.

## Explicit Non-Goals During The First Implementation Slice

The first coding slice may implement only the text backbone and text embedding
API because the other modalities depend on new processor and media contracts.
That sequencing must not remove image, audio, video, or multimodal fusion from
the overall definition of done.

No production implementation should begin until the text slice's tensor names,
configuration mapping, public embedding contract, and reference-fixture
strategy have been approved.
