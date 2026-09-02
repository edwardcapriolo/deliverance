# Shared KV Block Cache Plan

## Goal

Replace copy-based prefix-cache snapshots with KVCache2 shared immutable blocks.

The desired runtime model is:

```text
KvCacheManager
  owns shared block manager
  opens one KvCacheSession per query

KvCacheSession
  has a session id
  owns mutable tail blocks
  holds leases for shared immutable prefix blocks

KvBlockManager
  owns reusable immutable blocks
  tracks leases, refcounts, eviction, and optional disk persistence
```

This should make shared prefix caching a natural KVCache2 feature instead of a side cache that copies rows out and replays writes into a new session.

## Current State

KVCache2 already has most of the right internal seams:

- `KvCacheManager` creates request-local `KvCacheSession` instances.
- `KvCacheSession` tracks logical length and routes reads through committed and mutable blocks.
- `MutableKvBlock` owns writable dense rows for active writes.
- `MutableKvBlock.commit(...)` produces immutable `KvBlock` instances.
- `KvBlock` delegates physical representation to `KvBlockStorage`.
- `KvBlockStorage` already supports dense and `MSE_TURBOQUANT` committed layouts.
- `KvReadView` exposes logical read access to attention.

The missing piece is shared ownership. A committed `KvBlock` is immutable, but it is still owned by one session and closed when that session closes.

The current `KvPrefixSnapshotCache` should be treated as transitional. It stores copied dense prefix snapshots and restores them by replaying writes into a destination session. That proves the mechanical prefix-cache behavior, but it is not the long-term shared-block design.

## In Scope

### Exclusive Prefix Cache Modes

The copy-snapshot prefix cache and shared-block prefix cache are different systems and should not run at the same time.

Configuration should select one mode:

```java
KvBufferCacheSettings.withPrefixCacheMode(PrefixCacheMode.SNAPSHOT)
KvBufferCacheSettings.withPrefixCacheMode(PrefixCacheMode.SHARED_BLOCKS)
```

The existing `SNAPSHOT` mode keeps the old behavior:

- prefix entries are copied into snapshot storage
- checkpoint policies such as `START_AND_END` can store a selected set of prefixes, for example front blocks and back blocks when the checkpoint count is limited
- prefix compression settings apply to snapshot entries

The new `SHARED_BLOCKS` mode is block-manager based:

- full immutable KVCache2 blocks are retained by lease
- lookup attaches shared block leases instead of copying rows
- cache sizing is byte-budgeted by `sharedPrefixBlockCacheMaxBytes`
- snapshot compression settings do not apply

The settings object should make this mutually exclusive with one enum instead of independent enable flags. Existing snapshot lookup/store paths must no-op when `SHARED_BLOCKS` is selected. Shared-block lookup/admission must be wired separately and must not call into `KvPrefixSnapshotCache` or `KvBufferCache` snapshot storage.

### Per-Query Sessions

Every query opens a new `KvCacheSession` with a unique session id.

The session owns request-local state:

- current logical sequence length
- mutable tail/current blocks
- decode/prefill write cursor state
- a block table of attached immutable block leases
- release of all leases on close or crop

The session must not own reusable shared block memory. It owns handles to shared blocks.

### Concurrent Block Manager

Add a manager for shared immutable blocks:

```text
KvBlockManager
  ConcurrentHashMap<KvBlockKey, ManagedKvBlock>
```

`ManagedKvBlock` should contain:

- immutable `KvBlock`
- encoded byte size
- dense byte equivalent
- last access/update timestamp
- refcount or lease count
- eviction state
- optional disk residency metadata later

The block manager owns block lifetime. Sessions retain and release leases.

The block manager must be rank-aware.

In local mode there is one logical KV owner:

```text
tpSize=1
tpRank=0
block contains full-model KV for all local layers and all local KV heads
```

In tensor-parallel mode, blocks are rank-local:

```text
tpSize=N
tpRank=R
block contains only rank R's local KV shard
```

Rank-local means rank 0 must never attach rank 1's block, even if the token block hash is identical. The token-prefix identity can be common across ranks, but the KV payload is rank-specific.

If gossip reassigns ranks, a worker restarts a rank, or a node takes over a rank previously owned by another node, that rank must start with a new assignment epoch and must not reuse in-memory blocks from the old epoch. Disk/shared-storage reuse, if added later, needs exact rank/shard metadata validation and should remain separate from process-local in-memory reuse.

### Leases

Sessions should attach shared blocks through leases, not raw references.

Conceptual shape:

```java
record KvBlockLease(KvBlockKey key, KvBlock block, String sessionId) implements AutoCloseable
```

Correctness relies on the lease/refcount, not on scanning sessions. `sessionId` is useful for logging, metrics, leak detection, and debugging.

Required lifecycle:

```text
retain existing block
  atomically increment refcount if block is not evicting
  return lease

close session / crop prefix away
  close lease
  decrement refcount

evict block
  remove from lookup map
  mark evicting
  close immediately if refcount == 0
  otherwise close when final lease releases
```

A block may be:

- resident and unreferenced
- resident and referenced by active sessions
- evicted from lookup but still alive because active sessions hold leases
- closed after the final lease is released

### Shared Prefix Blocks

Only full committed blocks are shareable in the first implementation.

Mutable blocks are never shared. Partial tail blocks remain private to the session.

Prompt tokens should be split into fixed-size blocks. Each full token block gets a chained hash:

```text
blockHash = hash(parentBlockHash, tokenIdsInThisBlock)
```

The chained hash prevents unsafe reuse of the same local token chunk at a different prefix position.

The cache lookup should find the longest contiguous cached prefix from token block zero:

```text
tokens -> block hash chain -> retain each matching block -> attach leases -> prefill uncached suffix
```

### Cache Key

`KvBlockKey` must include every field that can change KV values or layout compatibility.

At minimum:

- model cache id or model weights fingerprint
- tokenizer id or tokenizer hash
- runtime/model salt
- layer index or all-layer block grouping decision
- block index / absolute token position range
- parent token block hash
- token block hash
- block size
- KV key dtype
- KV value dtype
- KV block storage layout, such as `DENSE` or `MSE_TURBOQUANT`
- TurboQuant bit width when applicable
- KV heads and head dimension
- RoPE/scaling config hash
- attention/sliding-window configuration

For tensor parallelism, include rank-local identity:

- tensor-parallel enabled flag
- deployment id
- tensor-parallel size
- rank id
- assignment epoch or generation id
- local shard range / local KV length
- Deliverance build/cache format version if rank workers may run different binaries

In local mode these fields should normalize to a single logical rank identity, for example `tpSize=1` and `tpRank=0`. That keeps one cache key shape for both local and tensor-parallel execution while making TP rank ownership explicit.

The first implementation should prefer exact matches only. Do not silently convert BF16 to I8, F32 to TurboQuant, or TurboQuant to dense as part of cache lookup.

The assignment epoch is important because ranks can move between nodes at runtime. `tpRank=1` today and `tpRank=1` after reassignment are not automatically the same cache owner. Reuse is safe only if the new rank has the same model files, same shard plan, same runtime KV settings, same code/storage format, and a deliberate epoch policy that allows lookup.

The rank fields are not optional metadata. They are part of correctness because tensor-parallel KV blocks contain only the local shard for that rank. Local mode is the degenerate case where the single rank owns the whole KV payload.

### Session Block Table

`KvCacheSession` should evolve from session-owned committed maps to a logical block table that can contain:

- owned mutable blocks
- owned committed blocks not admitted to the manager
- leased shared committed blocks

Attention should continue to read through `KvReadView`, page APIs, and row APIs. The goal is to change ownership underneath that boundary without redesigning model attention code.

If a rank assignment changes while sessions are active, the worker should close affected rank sessions and release their leases. The block manager may keep unreferenced blocks for the old epoch only if they are no longer reachable by new-epoch lookups. The safer v1 policy is to clear rank-local in-memory shared blocks when the local rank assignment changes.

### Admission

When a mutable block becomes full, commit it as today. Then optionally admit it to the block manager.

Conceptual flow:

```text
write rows into MutableKvBlock
advance length
full block commits to KvBlock
build KvBlockKey from token hash chain and runtime salt
admit block to KvBlockManager
session replaces owned committed block with retained lease if admitted
```

Admission should be byte-budgeted, not entry-count-only. TurboQuant blocks can still be multi-megabyte objects, so the cache must have an explicit memory ceiling before it is wired into model-level prefix reuse.

Initial policy can be simple:

- admit only full blocks
- admit only from prefixes within `maxPrefixTokensPerPrompt`
- evict by approximate LRU
- never evict a referenced block from memory until its leases release
- allow referenced blocks to temporarily exceed the resident byte budget rather than breaking active sessions
- evict the coldest unreferenced blocks when admission or release finds the cache over budget

The model/config knob is:

```java
KvBufferCacheSettings.withPrefixCacheMode(PrefixCacheMode.SHARED_BLOCKS)
KvBufferCacheSettings.withSharedPrefixBlockCacheMaxBytes(long maxBytes)
```

JSON config should map this through:

```json
{
  "kvBufferCache": {
    "prefixCacheMode": "SHARED_BLOCKS",
    "sharedPrefixBlockCacheMaxBytes": 536870912
  }
}
```

The default is intentionally finite so future wiring does not create an unbounded in-memory block cache by accident.

### Metrics

Add metrics from the first implementation.

Useful counters/meters/timers:

- `kvcache.v2.blockmanager.lookup`
- `kvcache.v2.blockmanager.hit.memory`
- `kvcache.v2.blockmanager.miss`
- `kvcache.v2.blockmanager.retain`
- `kvcache.v2.blockmanager.release`
- `kvcache.v2.blockmanager.admit`
- `kvcache.v2.blockmanager.evict`
- `kvcache.v2.prefix.tokens.reused`
- `kvcache.v2.prefix.bytes.attached`
- `kvcache.v2.block.seal.elapsed`
- `kvcache.v2.blockmanager.bytes.resident`
- `kvcache.v2.blockmanager.bytes.referenced`

The main success metric is reused tokens, not raw hit count.

## Out Of Scope For The First Shared-Block Work

### Active KV Spill To Disk

Do not make disk act as transparent swap for an active session.

This is out of scope:

```text
huge prompt exceeds RAM
  -> silently spill old active KV blocks to disk
  -> decode reads old KV blocks back from disk every token
```

Active decode attention repeatedly reads prior K/V. Disk reads in that path are likely catastrophic unless there is a separate paging scheduler, resident window policy, async prefetch, and explicit performance model.

If an active session cannot fit its required KV within the configured memory budget, the first implementation should fail clearly or require the caller to choose a smaller context, lower KV dtype, or different committed-block policy.

### Dtype/Layout Conversion During Lookup

Do not perform implicit cache-format conversion in v1.

Examples that should initially miss unless explicitly supported later:

- request wants I8 KV but disk has BF16 KV
- request wants dense BF16 but cache has `MSE_TURBOQUANT`
- request wants TurboQuant 3-bit but cache has TurboQuant 4-bit

Exact match first. Conversion can be a later explicit policy with dedicated metrics.

### Output Equivalence Claims

Do not claim generated text is always identical until split-prefill equivalence is verified for the target model, tensor provider, quantization mode, and attention implementation.

The mechanical invariant is required first:

```text
cached prefix blocks attach at the same absolute positions
uncached suffix prefill starts at the original suffix position
decode starts after the full prompt length
KV rows round-trip correctly
```

Output equivalence depends on model execution being chunk-invariant.

## Disk-Based Prefix Cache

Disk belongs behind the immutable block manager after in-memory shared blocks work.

Disk cache use cases:

- preserve expensive reusable prefixes across process restarts
- keep cold unreferenced shared prefix blocks outside memory
- support large agent/system prompts that repeat across sessions

Disk cache non-goals:

- not active session spill
- not hidden disk reads inside attention hot loops
- not reuse of existing active page files from `DiskKvBackend.md`

The existing disk KV backend stores active page files named by session/page coordinates. It is not durable prefix cache storage and should remain separate unless replaced by a later unified design.

Suggested disk layout:

```text
~/.deliverance/kv-cache/
  <model-cache-id>/
    tp-<size>/
      rank-<rank>/
        manifest.json
        blocks/
          <block-key-hash>.meta.json
          <block-key-hash>.bin
```

Each persisted block needs:

- cache format version
- full `KvBlockKey` fields or a verifiable serialized form
- checksum
- encoded byte length
- dense byte equivalent
- storage layout
- key/value dtype
- shape metadata
- block size and token count
- TurboQuant parameters if applicable
- tensor-parallel deployment/rank/shard metadata when TP is enabled

Writes should be atomic:

```text
write temp files
fsync data
fsync metadata/manifest as needed
rename into final path
```

Lookup order:

```text
memory block manager lookup
  hit -> retain and attach
disk lookup
  hit -> load or mmap into immutable KvBlockStorage -> admit/promote -> retain and attach
miss -> compute suffix normally
```

Disk reuse across rank movement should be conservative if disk persistence is local to a physical node. A node that newly owns rank `r` may load disk blocks for rank `r` only when metadata exactly matches the new rank context. In-memory blocks from a prior local rank assignment should not be reused across epochs in v1.

If a future deployment uses true shared storage, persistent block identity can be based on logical rank rather than physical node. That is a storage-tier decision. The core requirement remains the same: the block manager and cache key are rank-aware.

Disk metrics:

- `kvcache.v2.disk.lookup`
- `kvcache.v2.disk.hit`
- `kvcache.v2.disk.miss`
- `kvcache.v2.disk.load.elapsed`
- `kvcache.v2.disk.write.elapsed`
- `kvcache.v2.disk.bytes.read`
- `kvcache.v2.disk.bytes.written`
- `kvcache.v2.disk.evict`

### Initial Dense Disk Persistence Slice

The first disk implementation should be dense-only and opt-in. It is a correctness and measurement baseline, not the final local-disk recommendation.

Scope:

- persist exact `DENSE` committed `KvBlockStorage` only
- support dense `I8`, `BF16`, and `F32` key/value payloads
- preserve the exact chosen layout and dtype
- no TurboQuant serialization yet
- no layout or dtype conversion
- no active KV spill
- no synchronous disk writes on the generation hot path

Runtime mode:

```text
prefill computes block
commit immutable block
admit to memory block manager
generation continues
background writer persists block
```

Lookup mode:

```text
memory lookup
  hit -> attach lease
memory miss
  disk lookup/load before prefill
  validate metadata/checksum/exact key
  promote loaded block to memory manager
  attach lease
disk miss/corrupt/error
  recompute normally
```

Disk is a cold persistence tier behind memory. It is not a direct-disk-only KV cache.

Configuration:

```json
{
  "kvBufferCache": {
    "prefixCacheMode": "SHARED_BLOCKS",
    "sharedPrefixBlockCacheMaxBytes": 536870912,
    "sharedPrefixDiskCacheEnabled": true,
    "sharedPrefixDiskCachePath": "~/.deliverance/kv-cache",
    "sharedPrefixDiskCacheMaxBytes": 2147483648,
    "sharedPrefixDiskCacheReservedFreeBytes": 1073741824,
    "sharedPrefixDiskCacheMinUsableBytes": 1073741824,
    "sharedPrefixDiskCacheAdmitMinTokens": 256,
    "sharedPrefixDiskCacheWriterQueueSize": 128
  }
}
```

Safety gates:

- disk cache is disabled by default
- if `sharedPrefixDiskCacheMaxBytes` is below the minimum accepted disk budget, disable disk cache and record a metric
- if filesystem usable space is below `sharedPrefixDiskCacheMinUsableBytes`, disable disk cache and record a metric
- skip writes that would violate `sharedPrefixDiskCacheReservedFreeBytes`
- skip writes when the writer queue is full
- skip writes below `sharedPrefixDiskCacheAdmitMinTokens`
- skip unsupported layouts, including TurboQuant until exact encoded serialization is implemented
- never rewrite an existing block key
- corrupt or incomplete disk entries are treated as misses

Eviction:

```text
startup scan -> delete oldest entries until totalBytes <= maxBytes
after successful write -> delete oldest entries until totalBytes <= maxBytes
```

Disk eviction can delete files even if the same block is currently memory-resident, because active sessions hold memory leases. V1 disk eviction does not need active-session lease tracking.

Dense payload format:

```text
meta JSON: full key identity, dtype/layout/shape, payload checksum, payload bytes
bin: key rows then value rows in storage dtype byte order
```

Metrics:

- `kvcache.v2.disk.disabled.max_bytes_too_small`
- `kvcache.v2.disk.disabled.usable_bytes_too_small`
- `kvcache.v2.disk.lookup`
- `kvcache.v2.disk.hit`
- `kvcache.v2.disk.miss`
- `kvcache.v2.disk.load.elapsed`
- `kvcache.v2.disk.write.elapsed`
- `kvcache.v2.disk.bytes.read`
- `kvcache.v2.disk.bytes.written`
- `kvcache.v2.disk.write.skipped.low_space`
- `kvcache.v2.disk.write.skipped.queue_full`
- `kvcache.v2.disk.write.skipped.too_small`
- `kvcache.v2.disk.write.skipped.unsupported_layout`
- `kvcache.v2.disk.write.skipped.exists`
- `kvcache.v2.disk.evict`
- `kvcache.v2.disk.evict.bytes`

TurboQuant should follow only after dense proves the disk lifecycle. For TurboQuant, the rule remains exact-layout persistence: `MSE_TURBOQUANT` encoded payload to disk and back into `MSE_TURBOQUANT`, with no dense intermediate.

## Implementation Phases

### Phase 1: In-Memory Leased Blocks

Deliver a process-local shared prefix cache with no disk.

Tasks:

- add `KvBlockKey`
- add token block hash-chain utility
- add `KvBlockManager`
- add `ManagedKvBlock` and `KvBlockLease`
- teach `KvCacheSession` to attach leased committed blocks
- release leases on session close and crop
- include tensor-parallel rank and assignment epoch in cache identity
- clear rank-local in-memory cache on rank assignment changes
- keep `KvReadView` API stable
- add mechanical tests for attach/read/crop/close lifecycle

Success criteria:

- two sessions with the same full prefix share the same immutable blocks
- second session skips prefill for reused full blocks
- leases prevent close while a session is active
- evicted but referenced blocks remain readable until final release
- rank reassignment prevents stale rank-local cache reuse

### Phase 2: Replace Snapshot Prefix Cache

Replace `KvPrefixSnapshotCache` internals or delete it in favor of shared block cache wiring.

Tasks:

- preserve model-level prefix-cache behavior where practical
- stop snapshotting prefix rows into dense tensors
- stop restoring hits by replaying writes
- update `PrefixCache.md`
- remove or rewrite snapshot-specific tests

Current implementation status:

- `PrefixCacheMode.SHARED_BLOCKS` is wired into local KVCache2 generation.
- `AbstractModel` owns a `KvBlockManager` constructed from `sharedPrefixBlockCacheMaxBytes`.
- Local KVCache2 sessions look up shared blocks before prefill when shared-block mode is selected.
- Local KVCache2 sessions admit newly committed prompt blocks after suffix prefill.
- Snapshot mode remains the default and keeps the existing copy/restore prefix cache.
- Tensor-parallel generation is not wired to distributed shared-block lookup yet.

Success criteria:

- prefix hits attach block leases by reference
- no row-copy restore path on cache hit
- existing prefix-cache mechanical invariants still pass

### Phase 3: Byte-Budgeted Eviction

Add production cache sizing.

Tasks:

- track resident bytes and referenced bytes
- evict unreferenced blocks by approximate LRU
- keep referenced evicted blocks alive until release
- expose config in `KvBufferCacheSettings` or successor config object
- expose the same budget through model JSON config

Success criteria:

- cache remains under configured resident byte budget after admissions
- active sessions continue reading evicted blocks they already retained
- metrics show bytes resident and bytes referenced

### Phase 4: Disk Persistence

Add exact-format disk persistence for immutable blocks.

Tasks:

- define metadata and binary format version
- implement dense block serialization first
- implement TurboQuant block serialization after dense works
- add checksum validation
- add startup-safe directory handling and atomic writes
- add disk size budget and eviction

Success criteria:

- a cached prefix survives process restart
- exact metadata mismatch is treated as miss
- corrupted block fails closed and recomputes
- disk hit promotes to memory and attaches by lease

### Phase 5: Explicit Format Conversion Policies

Only after exact-format cache works, consider explicit conversions.

Possible policies:

- load BF16 and quantize to I8
- load dense F32 and encode to TurboQuant
- load TurboQuant and decode to dense for compatibility

Each conversion must be opt-in and profiled. The default should remain exact match.

## Open Questions

- Should a shared block contain all layers for a token block, or should the key include `layerIndex` and share per-layer blocks?
- What is the stable model fingerprint source for QOD artifacts, local Hugging Face downloads, and manually supplied model directories?
- Should cache salt include tensor provider/native-kernel identity, or only model math configuration?
- What is the first byte budget knob: shared block memory bytes, max reused tokens, or both?
- Should disk cache live under the model directory, under `~/.deliverance/kv-cache`, or be explicitly configured per model?
- How much of the current `KvBufferCacheSettings` prefix snapshot vocabulary should survive versus being renamed around shared blocks?
- Should exact output-equivalence tests be opt-in per model/provider after split-prefill equivalence is proven?
- What exact gossip/assignment value should become the tensor-parallel cache assignment epoch?
- Should disk cache reuse survive assignment epochs by default, or require an explicit operator setting after exact metadata validation?

## Recommendation

Start with Phase 1 and Phase 2 only.

That gives the important architecture:

```text
sessions own leases
block manager owns rank-aware immutable blocks
prefix hits attach blocks by reference
```

After that works, disk is a persistence tier for the same immutable block abstraction, not a separate cache design.

## Initial Implementation Decisions

The first implementation should use these locked decisions unless a test exposes a concrete problem:

- Scope is in-memory shared block leases only; no disk persistence in the first PR.
- Shared blocks are selected with `PrefixCacheMode.SHARED_BLOCKS`; they do not run alongside snapshot prefix caching.
- Local mode normalizes to `tpSize=1` and `tpRank=0`.
- Tensor-parallel mode requires rank-aware identity: `tpSize`, `tpRank`, assignment epoch, and local shard/KV identity.
- Process-local rank caches are cleared or made unreachable on rank reassignment.
- A shared `KvBlock` contains all local layers for one token block, matching the current `KvBlockStorage` shape.
- Do not split shared cache entries per layer in v1.
- Duplicate concurrent admissions for the same key are normal; the `putIfAbsent` winner owns the block.
- The duplicate loser closes its candidate block and retains the winner.
- The admission winner does not matter because `KvBlockKey` must include every correctness-relevant field.
- `KvPrefixSnapshotCache` is replaceable scaffolding; preserve user-facing behavior where practical, but remove copy/restore internals once shared blocks are wired.

The first tests should prove mechanical behavior before generated text:

- same prompt attaches leases to the same immutable blocks
- different suffix reuses only common full blocks
- partial blocks are not shared
- rank mismatch misses
- lease references protect blocks from close/eviction
- duplicate concurrent admits converge to one managed block
