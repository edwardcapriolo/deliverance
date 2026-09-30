# Qwen3 TensorRef Migration Analysis

## Context

Deliverance is migrating transformer execution from `AbstractTensor` and `TensorOperations` to `TensorRef` and
`Lighter`. The intended migration is a structural mirror of the existing implementation, changing only the tensor
library boundary.

The current `check-lightert2` branch instead changes several execution properties at once:

- tensor representation
- operation dispatch
- quantization boundaries
- attention implementation structure
- batching and parallelism
- KV cache access
- MLP fusion
- provider observability

The end-to-end symptom is that `Qwen06bBenchmarkCasesIT` is either slow or produces gibberish. This is not explained
by one defect. There are separate correctness and performance regressions.

## Verified Evidence

The focused legacy-versus-TensorRef test was run with:

```sh
MAVEN_OPTS="-XX:TieredStopAtLevel=1" \
  mvn -pl core -am \
  -Dtest=Qwen3HfTextModelPortedTest#tinyLegacyAndTensorRefExecutionMatchPrefillAndDecode \
  -Dsurefire.failIfNoSpecifiedTests=false test
```

It failed at the first shared attention boundary:

```text
0:attention_output max error=16.926018
```

This proves that layer 0 attention is already substantially different before MLP execution or accumulation across
later layers.

The focused unequal-offset GQA operation test was also run:

```sh
MAVEN_OPTS="-XX:TieredStopAtLevel=1" \
  mvn -pl native \
  -Dtest=Tensor2LighterDotProductRowsFuzzTest#f32DotProductRowsSupportsDifferentOperandOffsets test
```

It passed in the current worktree. This shows that the current native guard avoids the known unequal-offset error,
but the test permits fallback and therefore does not prove that the native provider performed the operation.

## Correctness Findings

### Native GQA Operand Offsets

GQA uses one query-head offset and a different KV-head offset:

```text
queryOffset = head * headSize
kvOffset    = (head / headGroupSize) * headSize
```

`KvCacheSelfAttention2` passes both offsets to `Lighter.dotProductRows`. The native F32 ABI used by
`NativeOps.dotProductRows` has only one shared column offset. Before the current uncommitted guard, native execution
read the wrong key columns for most GQA heads.

Relevant files:

- `core/src/main/java/io/teknek/deliverance/generator2/KvCacheSelfAttention2.java`
- `native/src/main/java/io/teknek/deliverance/tensor2/NativeOps.java`
- `native/src/test/java/io/teknek/deliverance/tensor2/Tensor2LighterDotProductRowsFuzzTest.java`

The current guard prevents corruption by rejecting unequal offsets, but it sends those operations to Panama. The
native `batchDotProduct` ABI already supports separate offsets and is a better production implementation target.

### Attention Output Projection Does Not Mirror Legacy

Legacy attention quantizes the attended values through `maybeQuantizeReadOnly` before applying `o_proj`.
TensorRef attention projects directly from the dense attended tensor.

Relevant boundaries:

- Legacy: `KvCacheSelfAttention.outputProjection`
- TensorRef: `KvCacheSelfAttention2.projectOutput`

This changes both numerical behavior and provider selection. It is a strong candidate for the remaining layer 0
attention mismatch, but it has not yet been isolated because the existing trace exposes only the post-projection
`attention_output` stage.

### Native Operations On TensorRef Slices

Attention passes sliced score and output rows to SAXPY. A `TensorRef` slice retains the parent's memory segment and
represents its logical base through `memorySegmentOffset`.

The current uncommitted native SAXPY changes rebase those segments before calling C. Without rebasing, a nonzero row
can read or update row zero. This can corrupt prefill attention. A dedicated sliced-SAXPY test is still missing.

Native dot-product methods have the same general slice-base risk and should be audited independently.

`PanamaOps.copySameDType` also copies from physical byte offset zero rather than the logical slice base, making
same-dtype reshape of a nonzero slice incorrect.

### Quantized KV Row Copying Is Incomplete

`BaseCausalSelfAttention2.copyRow` copies only packed payload bytes. I8 and Q4 tensors also have scale sidecars. A
payload-only copy is not a valid quantized tensor copy.

The legacy cache packing path allocates storage in the configured KV dtype and handles conversion. TensorRef packing
currently assumes matching dtypes and does not preserve quantization metadata.

### Tensor Parallel Semantics Are Missing

TensorRef attention returns the rank-local `o_proj` result without an all-reduce. `MLPBlock2` also lacks reduction and
uses global hidden dimensions with locally sharded weights.

The model boundary currently discards the supplied reducer when invoking TensorRef blocks. Multi-rank Qwen3 will
either fail shape validation or produce incomplete rank-local results.

This is not the immediate single-rank Qwen3-0.6B failure, but it means the new classes do not mirror the old classes.

### LoRA Semantics Are Missing

Legacy Q/K/V/O and MLP projections apply active LoRA deltas. No generator2 class consults `activeLoraDeltaFor`.
Qwen3 advertises LoRA hot-swap support, but TensorRef execution silently computes the base model.

### Residual Multiplier Is Applied To The Wrong Branch

Legacy residual behavior is:

```text
output = multiplier * sublayerOutput + residual
```

TensorRef currently performs:

```text
output = sublayerOutput + multiplier * residual
```

Normal Qwen3 checkpoints currently use a null multiplier, so this is latent for the benchmark model, but it is a real
semantic mismatch.

### TensorRef Blocks Are Effectively F32-Only

`CompositeOps.addResidual` and `CompositeOps.silu` require F32 tensors. Intermediate allocations use the configurable
working dtype. BF16 or F16 working execution can therefore fail even though the legacy implementation supports those
types through generic tensor access or provider operations.

### Looser Model Contracts

`MLPBlock2` hardcodes SiLU instead of honoring `config.activationFunction`. This matches the current Qwen3 checkpoint
but does not mirror the configurable legacy model contract.

## Performance Findings

### I8 QKV Input Falls To Scalar NaiveOps

The current uncommitted `TransformerBlock2` reshapes normalized attention input to `workingQType`, normally I8 for
the Qwen benchmark.

Tensor2 support is currently:

- native `dotProductRows`: F32/BF16 input
- Panama `dotProductRows`: F32 input
- Naive `dotProductRows`: generic scalar input

Therefore I8 activation times Q4 weights falls through to scalar `NaiveOps` for Q, K, and V projections. At real
Qwen dimensions this is a severe deterministic slowdown, not a minor tuning issue.

Legacy has optimized I8/Q4 projection kernels. A faithful migration must implement the equivalent Lighter operation
rather than changing the quantization boundary or relying on scalar fallback.

### Decode Copies The Entire KV History

For every decode token and every layer, `KvCacheSelfAttention2` allocates packed K/V tensors, copies the full visible
prefix, appends the current row, performs attention, and releases the buffers.

Legacy decode uses page-backed attention directly. The TensorRef path adds O(context length) copying on top of the
attention calculation for every generated token.

This structural difference does not explain the current prefill parity failure. The failing parity test uses
`startPosition == 0`, where TensorRef uses current K/V directly and performs no prefix packing. It remains a major
decode-performance and quantized-KV correctness problem.

### Prefill Lost Packed Block Parallelism

Legacy `PackedBlockAttention`:

- partitions heads through the model pool
- processes query rows in bounded blocks
- limits concurrent score memory
- uses `Lighter.batchDotProduct`
- supports separate query and KV offsets

TensorRef attention loops heads serially and calls `dotProductRows` over the complete score matrix for each head.
Even when mathematically correct, this is not a tensor-library-only rewrite of the execution structure.

### MLP Lost Quantization, Fusion, And Parallelism

Legacy MLP behavior includes:

- quantized projection input
- chunked output rows
- phase-specific provider selection
- gate/up projection parallelism
- fused activation, multiply, and quantization where supported
- quantized down-projection input
- tensor-parallel handling

`MLPBlock2` performs separate gate and up projections, a standalone SiLU pass, a standalone multiply pass, and a dense
down projection. This increases memory traffic and can select weaker kernels.

### Provider Selection Is Opaque

`Lighter.dotProductRows` and SAXPY do not record which provider succeeds. Benchmark provider options configure the
legacy `TensorOperations` stack but do not pin the independent `Lighter` registry.

A benchmark can report a native legacy provider while TensorRef silently falls from native to Panama or Naive. This
makes current benchmark labels insufficient for diagnosing TensorRef performance.

## Why Existing Tests Do Not Isolate The Problem

### Tiny Legacy/TensorRef Test

The current test is useful because it proves a large first-layer divergence, but it has important weaknesses:

- It compares only stage names present in both traces.
- It does not require identical stage sets.
- It does not assert that trace maps are nonempty.
- Its stage tolerance is `0.1`.
- It observes attention only after output projection.
- It does not trace decode stages.
- It uses tiny dimensions that do not represent production kernels.
- Legacy uses an explicitly configured naive provider while TensorRef uses an independent provider chain.
- It does not assert which TensorRef provider executed.

### Qwen06bBenchmarkCasesIT

The integration test asserts only that the response is not blank. Gibberish satisfies that assertion. It does not
check logits, token IDs, a response digest, provider selection, or minimum performance.

### Native GQA Test

The current GQA offset test compares final results through a `Lighter` that includes fallback providers. Its SIMD
candidate can pass because native rejects the operation and Panama computes it. This proves fallback-chain
correctness, not native kernel correctness or performance.

## Required Parity Test Layers

### Primitive Operation Tests

Construct strict provider-specific `Lighter` instances with no fallback providers:

- Naive-only oracle
- Panama-only production baseline
- SIMD-only native path

Cover:

- F32/F32, F32/Q4, BF16/Q4, and I8/Q4
- unequal GQA operand offsets
- nonzero tensor slices
- row and output offsets
- SIMD tails and odd dimensions
- Q4 and I8 block boundaries
- quantization scale sidecars

An unsupported production combination should fail the test rather than silently execute Naive code.

### Attention Stage Parity

Expose and compare identical legacy and TensorRef stages:

1. normalized attention input
2. quantized QKV projection input
3. query projection
4. key projection
5. value projection
6. normalized query
7. normalized key
8. rotated query
9. rotated key
10. raw attention scores per head
11. scaled/softcapped probabilities
12. attended value output
13. quantized output-projection input
14. projected attention output

The assertion should report the first differing layer, stage, row, head, and column. Stage-key sets, shapes, and dtypes
must match exactly.

### Block And MLP Parity

Compare:

1. post-attention residual
2. post-attention normalization
3. quantized MLP input
4. gate projection
5. up projection
6. activated gate
7. multiplied gate/up result
8. quantized down-projection input
9. down projection
10. reduced MLP output
11. post-FF residual

Include a non-unit residual multiplier test and configured activation coverage.

### KV Cache Parity

After every prefill and decode step, compare logical key/value rows between paths for:

- F32 KV
- BF16 KV
- I8 KV
- GQA layouts
- block boundaries
- multiple decode steps

For quantized formats, compare both dequantized values and scale sidecars.

### Real Model Parity

Use the cached Qwen3-0.6B checkpoint with fixed prompt and continuation token IDs. Compare:

- every layer output or selected checkpoint layers
- final normalized hidden state
- complete logits or a stable digest
- top-k token IDs and scores
- greedy token ID at every step
- exact generated token sequence

Generated prose should be diagnostic output, not the primary correctness assertion.

### Performance Gates

Measure prefill and decode separately. Assert:

- no Naive provider calls on production shapes
- no full-history KV copy during ordinary decode
- batched/parallel QKV projection
- batched/parallel prefill attention
- fused or equivalently efficient MLP execution
- throughput near the known-good legacy baseline

## Remediation Plan

### Phase 1: Build The Parity Harness

Add exact stage hooks and strict provider-specific operation tests before changing production behavior. Keep legacy and
TensorRef invocation as test fixtures only. Do not add a production runtime switch or fallback flag.

The first goal is to turn the current `attention_output` mismatch into the exact first failing operation.

### Phase 2: Restore Attention Semantics

Mirror the legacy order and boundaries exactly:

1. pre-attention normalization
2. working-type quantization
3. Q/K/V projection
4. LoRA deltas
5. Q/K per-head RMSNorm
6. RoPE
7. KV write
8. attention score/value calculation
9. attended-value quantization
10. output projection
11. LoRA output delta
12. tensor-parallel reduction
13. reducer callback

Implement TensorRef equivalents of packed block prefill and page-backed decode rather than dense-history copying.

### Phase 3: Complete Production Kernels

Implement optimized Tensor2 operations required by the mirrored execution:

- I8/Q4 projection in Panama and native providers
- GQA scoring with distinct operand offsets
- slice-safe native dot products and SAXPY
- sidecar-aware quantized copies
- batched row projections with realistic tiling

Provider support should be explicit. Naive remains a reference implementation, not a production rescue path.

### Phase 4: Restore MLP Semantics And Performance

Mirror legacy input quantization, configured activation, gate/up parallelism, activation/multiply/quantize fusion,
down projection, LoRA, and tensor-parallel reduction using TensorRef/Lighter operations.

### Phase 5: Add Provider Observability

Record attempted and selected providers for hot operations with dtype and shape tags. Add counters for fallback reasons
and a test invariant that real Qwen prefill/decode performs zero Naive hot-path operations.

### Phase 6: Validate The Real Model

Run deterministic real-model logits/token parity first. Once correctness passes, run separate prefill and decode
performance gates against the known-good legacy baseline.

### Phase 7: Complete The Migration

Remove the legacy transformer execution path only after:

- primitive provider parity passes
- stage-by-stage block parity passes
- KV cache parity passes
- real-model logits and token parity pass
- provider assertions show no scalar production fallback
- performance gates pass

The finished design should not depend on flags that turn TensorRef execution or individual optimizations off.

## Current Working Tree Notes

The current uncommitted changes include useful evidence and partial corrections:

- unequal-offset native F32 rejection
- native SAXPY slice rebasing
- attention softcap propagation
- avoiding redundant K/V packing for `startPosition == 0`
- an initial legacy/TensorRef parity test

They do not restore complete parity. In particular, the current I8 attention-input conversion guarantees scalar QKV
projection with the available Tensor2 provider support, and the focused model parity test still fails at layer 0
attention output.
