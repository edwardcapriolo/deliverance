# KV TurboQuant Plan

## Goal

Implement TurboQuant as a first-class KVCache2 committed-block storage layout, not as a prefix-cache side channel and not as a model-specific attention hack.

KVCache2 should make this natural:

- `KvCacheSession` owns logical request cache state.
- `MutableKvBlock` owns writable dense rows for the active append/block.
- `KvBlock` owns immutable committed physical storage.
- `KvReadView` exposes immutable committed blocks plus the mutable active tail.
- Attention consumes `KvReadView` or a block/page reader interface.
- Compression/quantization is a committed block storage layout detail.

## Current Starting Point

The existing TurboQuant code is `MseTurboQuantCodec` and is used only by `KvBufferCache` prefix snapshots.

Current behavior:

- prefix cache stores compressed immutable snapshots
- lookup hydrates compressed rows back into a normal `KvBuffer`
- live attention still reads dense KV
- docs explicitly say this is not live KV-cache compression

That implementation already contains useful mechanics:

- row L2 norm
- deterministic sign vector
- Walsh-Hadamard rotation
- Lloyd-Max scalar codebook
- bit packing/unpacking
- encode/decode tests
- prefix-cache metrics proving large compression, roughly `90 MB -> 10-12 MB` in prior Qwen3-4B-JQ4 measurements

But it must not be copied into a second scalar hot path. KV TurboQuant should reuse the codec semantics while moving hot encode/decode/dot-product work behind tensor/SIMD/provider primitives where needed.

## Non-Goals

- Do not add Java scalar loops to production attention hot paths and call that the final implementation.
- Do not hydrate whole compressed KV blocks back into dense tensors before attention unless it is a temporary baseline benchmark.
- Do not put TurboQuant-specific branching into model classes.
- Do not make prefix cache own live KV layout decisions.
- Do not assume the current MSE reconstruction codec is full TurboQuant from the paper. It is not the QJL inner-product variant.

## Target Design

Introduce a storage-layout boundary under `KvBlock`.

Conceptual shape:

```text
MutableKvBlock
  dense writable storage
  append/write only

commit()
  applies configured committed-block policy

KvBlock
  immutable metadata
  KvBlockStorage storage

KvBlockStorage
  DenseKvBlockStorage
  MseTurboQuantKvBlockStorage
  future Q8/Q4/F16/BF16 storage

KvReadView
  exposes logical visible blocks/rows
  does not expose writable cache storage

PackedBlockAttention or replacement
  consumes KvReadView/block readers
  dispatches to dense or TurboQuant-aware kernels
```

The desired user-facing model knob should be builder/config-level, for example:

```java
.withKvBlockStorage(KvBlockStoragePolicy.MSE_TURBOQUANT)
```

or, if we keep settings consolidated:

```java
.withKvBufferCacheSettings(new KvBufferCacheSettings(true)
    .withKvBlockStoragePolicy(MSE_TURBOQUANT)
    .withKvTurboQuantBits(4))
```

Naming can change, but the policy belongs to KVCache2 committed block storage, not to prefix-cache compression.

## Storage API

Add an internal block storage abstraction. Exact names can vary, but it should express these capabilities:

```java
interface KvBlockStorage extends AutoCloseable {
    KvBlockLayout layout();
    DType denseDType();
    int layers();
    int tokenCount();
    int kvLength();

    AbstractTensor keyRowView(int layer, int blockRow);
    AbstractTensor valueRowView(int layer, int blockRow);

    void copyKeyRow(int layer, int blockRow, AbstractTensor destination);
    void copyValueRow(int layer, int blockRow, AbstractTensor destination);
}
```

Initial layouts:

```java
enum KvBlockLayout {
    DENSE,
    MSE_TURBOQUANT
}
```

`DenseKvBlockStorage` wraps the current `[layers, 2, blockSize, kvLength]` tensor.

`MseTurboQuantKvBlockStorage` owns encoded rows:

- row order: `((layer * blockSize) + blockRow) * 2 + keyOrValue`
- packed codes
- norms
- bit width
- `kvLength`
- `rotatedDim`
- immutable after construction

For compatibility, `keyRowView`/`valueRowView` for TurboQuant may initially return a decoded scratch-backed read-only tensor owned by the returned view. That is acceptable as a baseline if it is explicit and measured. The performance target is for attention to avoid per-row object churn and decode directly into reusable scratch/tile buffers.

## Commit Path

`MutableKvBlock.commit(tokenCount)` should not always produce a tensor-backed `KvBlock`.

Instead:

1. Validate all required rows are written.
2. Consult a committed-block storage policy.
3. For `DENSE`, transfer current dense storage into `DenseKvBlockStorage`.
4. For `MSE_TURBOQUANT`, encode dense rows into `MseTurboQuantKvBlockStorage` and close/release dense mutable storage.
5. Return immutable `KvBlock` containing metadata plus storage.

Only committed full blocks should be compressed initially. Mutable active/tail blocks stay dense because they are still being written, cropped, or verified.

Partial tail handling:

- Keep partial tail blocks dense.
- If a partial block later fills and commits, then compress it.
- If `crop()` splits a committed TurboQuant block, decode only the retained rows into a new dense mutable tail. This path is not the hot attention path and can prioritize correctness.

## Attention Path

Current `PackedBlockAttention` signature takes dense tensors:

```java
forward(output, query, keys, values, prefixRows, queryRows, ...)
```

That forces packing/hydration before attention. The target signature should consume a read view or reader:

```java
forward(output, query, KvReadView readView, AbstractTensor currentKeys, AbstractTensor currentValues, ...)
```

or:

```java
forward(output, query, KvAttentionSource source, ...)
```

where `KvAttentionSource` exposes logical rows/pages without requiring a dense packed tensor.

Important: this seam is required before TurboQuant can be a clean implementation. If TurboQuant is implemented by making `KvCacheSelfAttention` manually decode compressed rows into dense `packedKeys`, the design has regressed to a workaround.

## TurboQuant Attention Strategy

There are two milestones.

### Milestone 1: Correct Compressed Committed Blocks

Implement TurboQuant block storage and decode-on-read.

Purpose:

- prove KVCache2 lifecycle works with compressed immutable blocks
- measure memory traffic reduction
- measure output drift
- preserve existing attention math by decoding rows/tiles into temporary dense scratch

This may or may not be faster. It is valuable if memory bandwidth dominates enough that `10-12 MB` compressed reads beat `90 MB` dense reads plus decode cost.

### Milestone 2: TurboQuant-Aware Attention Kernels

Avoid reconstructing full rows unnecessarily.

For score computation:

- use provider/SIMD primitive to decode or dot a TurboQuant row/tile against a query head
- reuse scratch buffers per thread/head
- avoid object allocation per row
- count compressed bytes read and decoded coordinates

For value accumulation:

- decode value rows/tile into scratch and accumulate with `saxpy`-equivalent provider primitive
- later evaluate whether value rows need the same quantization as key rows

This is where speed should come from. The goal is lower memory bandwidth and fewer dense KV reads, not simply smaller storage.

## SIMD And TensorOperations Requirements

Do not implement production TurboQuant KV attention with scalar Java loops in `PackedBlockAttention`.

Acceptable short baseline:

- use existing `MseTurboQuantCodec` decode mechanics for correctness tests and one benchmark gate
- mark scalar decode as baseline/temporary if it is on the attention path

Production path should add or reuse provider primitives such as:

```java
decodeTurboQuantRow(...)
turboQuantDotProduct(...)
turboQuantSaxpy(...)
turboQuantScoreValue(...)
```

The exact primitive boundary should be selected after inspecting existing `TensorOperations` and native SIMD capabilities. Prefer a fused primitive if measurements show decode plus dot plus saxpy spends time in memory movement or allocation.

Likely progression:

1. Java reference under tests only.
2. Panama/vector provider implementation for correctness and portable performance.
3. Native SIMD implementation if Panama cannot hit target throughput.
4. GPU support only after CPU semantics are stable.

## Reusing Existing Prefix TurboQuant

Refactor rather than duplicate.

The existing `MseTurboQuantCodec` should become a reusable codec component for both:

- old prefix-cache snapshot compression
- new KVCache2 committed-block storage

Possible refactor:

- move metric names out of codec or pass a metric prefix
- keep codebook/sign caches shared
- keep encode/decode math shared
- add provider-backed encode/decode hooks where available
- preserve existing prefix-cache tests while adding KVCache2 tests

Metric names should distinguish old prefix cache from live KV:

- `kvbuffercache.prefix.turboquant.*` remains for legacy prefix snapshots
- `kvcache.v2.turboquant.*` for committed-block storage
- `packedblockattention.turboquant.*` for attention consumption

## Correctness Tests

Unit tests:

- dense block commit preserves existing behavior
- TurboQuant block commit produces immutable compressed storage
- TurboQuant block row decode approximately matches dense source rows
- `KvReadView.keyRow/valueRow` work for dense and TurboQuant committed blocks
- `TrackedReadOnlyTensor` still catches mutation for dense row views
- crop of a TurboQuant committed block decodes retained rows into dense mutable tail
- no-update denoise reads compressed committed prefix without writing it

Attention tests:

- dense KV attention and TurboQuant KV attention produce bounded-error outputs on small deterministic tensors
- causal mask and bidirectional block visibility match dense baseline
- GQA head grouping matches dense baseline
- softcap path matches dense baseline within tolerance

Integration tests:

- Nemotron instruct diffusion smoke with `trackKvReadViews=true` and TurboQuant committed KV enabled
- compare generated text is non-empty and no checksum assertion fires
- compare NFE/accepted-token counters against dense baseline to detect severe quality drift

## Benchmarks And Metrics

Profile counters/timers must make the tradeoff visible:

- `kvcache.v2.turboquant.block.encode`
- `kvcache.v2.turboquant.block.decode.row`
- `kvcache.v2.turboquant.encoded.bytes`
- `kvcache.v2.turboquant.dense.bytes.equivalent`
- `packedblockattention.turboquant.score`
- `packedblockattention.turboquant.value`
- `packedblockattention.score_value`
- existing `kvcacheselfattention.pack_kv`

Benchmark comparisons:

- dense KV, tracking off
- dense KV, tracking on
- TurboQuant KV, scalar baseline if implemented
- TurboQuant KV, provider/SIMD path

For Nemotron diffusion, always record:

- total wall time
- generated tokens
- NFE
- accepted tokens per block
- `packedblockattention.score_value`
- `kvcacheselfattention.pack_kv`
- logits projection time
- output text sanity

Quality drift can change NFE and accepted-token counts, so speed comparisons must include both timing and generation dynamics.

## Implementation Order

1. Add `KvBlockStorage` and convert current `KvBlock` to use `DenseKvBlockStorage` with no behavior change.
2. Move `MutableKvBlock.commit()` to produce storage through a policy object.
3. Add `KvReadView` block/page iteration APIs so attention can consume committed blocks without dense repacking.
4. Refactor `PackedBlockAttention` to consume the new read interface for dense blocks first.
5. Refactor `MseTurboQuantCodec` for reusable metric prefixes and block-row ordering.
6. Add `MseTurboQuantKvBlockStorage` with decode-on-read correctness tests.
7. Run Nemotron instruct diffusion with TurboQuant committed KV and tracking enabled.
8. Add provider/SIMD TurboQuant decode/dot/value primitives for the attention hot path.
9. Rebenchmark and decide whether FlashAttention-style tiling is still the next highest-value step.

## Decision Criteria

TurboQuant KV is worth keeping if it provides one or more of:

- clear memory reduction without unacceptable quality drift
- lower `packedblockattention.score_value` time
- better long-context capacity
- lower allocator pressure and fewer dense KV bytes moved

If decode overhead dominates and attention time does not improve, keep the storage abstraction but prefer exact lower-precision KV formats such as BF16/F16/Q8 before further TurboQuant work.
