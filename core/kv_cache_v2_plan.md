# KV Cache v2 Plan

## Goal

Replace the current mutable `KvBufferCache.KvBuffer`-centric design with a cache model that maps cleanly to upstream HF `Cache`/`DynamicCache`, vLLM-style block caching, flash-attention page tables, TurboQuant prefix snapshots, and diffusion decoding.

The practical goal is to stop re-solving cache semantics per model. New model ports should be able to express:

- AR prefill and cached decode.
- Split prefill and prefix cache reuse.
- Immutable block-level prefix sharing.
- Flash-attention-style paged reads.
- TurboQuant or other compressed prefix snapshots.
- Diffusion active-block decoding against a cached causal prefix.
- Linear self-speculation draft/verify cache cropping.

## Current Problem

Current `KvBufferCache` mixes several responsibilities:

- Active per-request mutable KV storage.
- Prefix lookup and snapshot storage.
- Prefix copy/hydration.
- Disk active-page allocation and cleanup.
- Compression codecs such as LZ4 and MSE TurboQuant.
- Page-table views for attention.
- Current context-position tracking.

This makes model ports fragile. A model that needs a slightly different upstream cache mode, such as Nemotron diffusion, ends up bypassing the shared generation stack instead of composing with it.

## Upstream Semantics To Match

### HF DynamicCache

HF cache semantics used by Nemotron:

- Cache is per layer.
- Each layer has key and value tensors.
- Prefill updates cache with prompt K/V.
- Decode appends one or more positions.
- `cache_position` is explicit and absolute.
- `position_ids` derive from `cache_position` unless supplied.
- Denoising can read cached prefix K/V without updating cache.
- Post-block causal forward updates cache after a diffusion block is accepted.
- Linear self-speculation can crop cache back to an accepted prefix.

### vLLM-Style Block Cache

The design we want from vLLM-style prefix caching:

- KV is stored in fixed-size token blocks.
- Full blocks are immutable after commit.
- Prefix cache entries refer to immutable blocks, not copies of mutable buffers.
- Cache hits attach existing blocks by reference when dtype/provider/layout match.
- A mutable tail exists only for the currently active request/block.
- Cache keys are token-prefix block hashes plus model/runtime salt.

## Proposed Abstractions

### `KvCacheSession`

Owns one request's logical cache state.

Responsibilities:

- Track logical sequence length.
- Track committed immutable blocks.
- Track one mutable append block.
- Provide per-layer write handles for prefill/decode.
- Provide read views for attention.
- Support `crop(int newLength)` for speculative decoding.
- Support `forkReadOnlyPrefix(int prefixLength)` for prefix reuse.

Not responsible for:

- Compression policy.
- Prefix lookup policy.
- Model-specific RoPE/mask behavior.

### `KvBlock`

Immutable committed KV block.

Fields/concepts:

- `blockId`
- `modelFingerprint`
- `dtype`
- `layout`
- `blockSize`
- `tokenCount`
- `tokenHash`
- per-layer key/value page references
- optional compression metadata

Rules:

- No writes after commit.
- Reference counted or owned by cache manager.
- Safe to share across sessions.
- Can be backed by heap, mmap, native memory, GPU memory, or compressed storage.

### `MutableKvBlock`

Append-only block for active generation.

Responsibilities:

- Accept K/V writes at explicit absolute positions.
- Validate monotonic append unless explicitly opened in random-write mode for tests/import.
- Commit to `KvBlock` only when full or when explicitly snapshotting final prefix.
- Expose read view that includes uncommitted rows for the active request.

### `KvReadView`

Immutable attention-facing view.

Responsibilities:

- Represents visible K/V for a layer and request step.
- Provides page table for flash attention or current paged attention.
- Can be composed from committed prefix blocks plus mutable active block.
- Carries logical `visibleTokens`, `cachePosition`, and mask/pattern metadata.

Important: attention should read `KvReadView`, not mutate the cache directly.

### `KvWriteCursor`

Explicit write API for model layers.

Responsibilities:

- Write key/value rows for `layer`, `absolutePosition`, and optional batch row.
- Validate dtype/layout.
- Return committed block events when an append crosses block boundary.

This replaces direct use of `getKeyTensorForPosition(...).copyFrom(...)` in model code.

### `PrefixBlockCache`

Maps token-prefix block keys to immutable `KvBlock` references.

Responsibilities:

- Lookup longest block-aligned prefix.
- Store committed full prompt blocks.
- Evict by blocks, bytes, recency, or model salt.
- Never mutate returned blocks.

This replaces copy-first prefix snapshots as the primary design. Copy/hydration remains a fallback only when layout/provider differs.

### `KvCompressionCodec`

Compression boundary for prefix blocks.

Implementations:

- `NoneKvCompressionCodec`
- `Lz4KvCompressionCodec`
- `MseTurboQuantKvCompressionCodec`

Responsibilities:

- Encode immutable committed blocks or block groups.
- Decode into immutable block storage or an attachable read-only backing.
- Report reconstruction error metrics.
- Refuse unsafe layouts/dtypes explicitly.

TurboQuant should compress immutable committed blocks only, not live mutable blocks.

### `AttentionPattern`

Describes visibility, independent of cache storage.

Examples:

- `CAUSAL`
- `BIDIRECTIONAL_BLOCK`
- `PREFIX_CAUSAL_PLUS_BIDIRECTIONAL_BLOCK`
- `SLIDING_WINDOW`
- `BLOCK_DIFFUSION_FLEX`

For Nemotron diffusion, the important pattern is:

- cached prefix visible to all block tokens
- active block tokens visible bidirectionally inside the block during denoising
- no cache update during denoising
- causal post-block update after block acceptance

### `CacheExecutionMode`

Explicit model-forward modes:

- `PREFILL_UPDATE_CACHE`
- `DECODE_UPDATE_CACHE`
- `READ_PREFIX_NO_UPDATE`
- `DENOISE_BLOCK_NO_UPDATE`
- `VERIFY_AND_UPDATE_CACHE`

This avoids hidden booleans like `use_cache=false` meaning different things in different models.

## How This Supports Nemotron Diffusion

Upstream algorithm maps cleanly:

1. AR causal prefill:
   - mode: `PREFILL_UPDATE_CACHE`
   - writes prompt blocks
   - samples `next_token`

2. Start diffusion block:
   - create active token block of mask tokens
   - first token seeded from AR if `causal_context=true`

3. Denoising step:
   - mode: `DENOISE_BLOCK_NO_UPDATE`
   - read view: committed prefix + active block
   - attention pattern: `PREFIX_CAUSAL_PLUS_BIDIRECTIONAL_BLOCK`
   - logits only for masked block positions
   - update token block, not KV cache

4. Post-block causal update:
   - mode: `VERIFY_AND_UPDATE_CACHE`
   - write accepted block K/V into cache
   - commit full KV blocks
   - sample seed for next block

5. Linear self-speculation:
   - draft block with no cache update
   - verify causally with cache update
   - crop cache to accepted length

## How This Supports Flash Attention

Flash attention needs a stable read-only view of K/V pages:

- `KvReadView` exposes page table metadata.
- Committed `KvBlock`s are immutable and can be shared safely.
- Mutable active block can be included as a final page segment.
- Attention kernels do not call mutable `getKeyTensorForPosition` APIs.
- Page layout can be chosen around kernel needs rather than Java object convenience.

## How This Supports TurboQuant

TurboQuant should become a block-storage codec, not a side behavior of prefix snapshot maps.

Policy:

- Compress only immutable committed prefix blocks.
- Store codec metadata alongside block identity.
- On lookup, either attach compressed block through a decoder-backed read view or hydrate into immutable blocks.
- Measure row error and generation drift at the block boundary.
- Never compress mutable active decode/denoise blocks.

## Minimum Interfaces

Sketch only. Names can change during implementation.

```java
interface KvCacheSession extends AutoCloseable {
    int length();
    KvWriteCursor writer(CacheExecutionMode mode);
    KvReadView readView(int layer, int visibleTokens, AttentionPattern pattern);
    void appendToken(int tokenId);
    void crop(int newLength);
    List<KvBlock> committedBlocks();
}

interface KvWriteCursor extends AutoCloseable {
    void write(int layer, int absolutePosition, AbstractTensor key, AbstractTensor value);
}

interface KvReadView extends AutoCloseable {
    int layer();
    int visibleTokens();
    AttentionPattern pattern();
    KvPageTable pageTable();
}

interface PrefixBlockCache {
    PrefixHit lookup(int[] tokens, Optional<String> salt, CacheFingerprint fingerprint);
    void store(int[] tokens, List<KvBlock> committedBlocks, Optional<String> salt, CacheFingerprint fingerprint);
}

interface KvCompressionCodec {
    EncodedKvBlock encode(KvBlock block);
    KvBlock decode(EncodedKvBlock encoded, KvBlockAllocator allocator);
}
```

## Migration Plan

### Implemented: Standalone v2 Core Slice

- Added standalone v2 package `io.teknek.deliverance.tensor.kv`.
- Added `KvCacheManager`, `KvCacheSession`, `KvWriteCursor`, `KvReadView`, `KvBlock`, `MutableKvBlock`, `AttentionPattern`, and `CacheExecutionMode`.
- `AbstractModel` now owns a side-by-side `KvCacheManager` and exposes `kvCacheManager()` / `newKvCacheSession()`.
- v2 storage is real block storage, not a wrapper/delegate over `KvBufferCache`.
- Full blocks commit to immutable `KvBlock` instances.
- Mutable active blocks remain writable until committed.
- `DENOISE_BLOCK_NO_UPDATE` write cursors reject KV writes.
- Current minimal `crop` supports block-boundary committed-block cropping and mutable-tail cropping; splitting an immutable block is explicitly rejected until speculative partial-block support is implemented.
- Focused tests: `KvCacheSessionTest` covers write/read, immutable commit behavior, read-view visible row copies, no-update write rejection, and mutable-tail crop behavior.

### Step 1: Nemotron AR On v2 Cache

- Add a Nemotron-specific AR backend that uses `KvCacheSession` directly.
- Causal prefill writes K/V through `KvWriteCursor` and commits full blocks.
- Decode reads through `KvReadView` and appends one token without recomputing the full sequence.
- Keep existing `KvBufferCache` and other model backends untouched.

### Step 2: Nemotron Diffusion Cached Prefix

- Implement upstream Nemotron diffusion over `KvCacheSession`.
- Causal prefill once.
- Denoise active block with `DENOISE_BLOCK_NO_UPDATE` and `PREFIX_CAUSAL_PLUS_BIDIRECTIONAL_BLOCK` read views.
- Post-block causal update writes accepted block K/V and commits full blocks.

### Step 3: Partial-Block Crop For Self-Speculation

- Extend `crop` to split or rebuild a partially committed block safely.
- Required for linear self-speculation acceptance lengths that stop inside a block.

### Step 4: Prefix Block Cache

- Add a v2 prefix cache keyed by token-prefix block hashes.
- Store and attach immutable `KvBlock` references.
- Do not copy mutable buffers as the primary path.

### Step 5: Compression Codecs On Blocks

- Move LZ4 and MSE TurboQuant behind `KvCompressionCodec`.
- Add block reconstruction error metrics.
- Add tests that compressed blocks hydrate to equal/close row values.

### Step 6: Attention Pattern API

- Make attention consume `KvReadView` and `AttentionPattern`.
- Causal AR uses `CAUSAL`.
- Diffusion uses `PREFIX_CAUSAL_PLUS_BIDIRECTIONAL_BLOCK`.

### Step 7: Flash Attention Integration

- Make `KvReadView` expose contiguous/page-table metadata needed by native flash attention.
- Add a provider path that can consume immutable prefix pages plus mutable tail.

## Tests Required

- Immutable block cannot be written after commit.
- Prefix lookup returns shared immutable blocks, not mutable copies.
- Cache hit attaches blocks and preserves decode start position.
- Copy fallback preserves old behavior when layout/provider differs.
- `crop` removes speculative tail without mutating shared prefix blocks.
- TurboQuant codec reports reconstruction error and never mutates source block.
- Flash/page read view lists pages in correct logical order.
- Nemotron diffusion denoise reads prefix without updating cache.
- Nemotron post-block update appends accepted block and advances length.

## Non-Goals

- Do not make every model port implement its own cache class.
- Do not let attention kernels own cache mutation.
- Do not compress mutable active blocks.
- Do not claim prefix-cache output equivalence until split-prefill/cache semantics are tested per model/provider.
