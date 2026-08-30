# KVCache2 Roadmap

## Current State

KVCache2 is request-local and in memory.

- `MutableKvBlock` owns writable active rows.
- `KvBlock` owns immutable committed storage.
- `KvBlockStorage` supports dense and `MSE_TURBOQUANT` committed layouts.
- `KvReadView` exposes read access for attention.
- `TrackedReadOnlyTensor` can assert that borrowed non-copying views are not mutated.

There is not yet a global KVCache2 prefix cache, shared block table, or disk backend.

## 1. Configurable KV2 Key/Value Storage Sizes

Add explicit KV cache storage policies for keys and values instead of deriving dense KV storage only from `workingMemoryType`.

Target capabilities:

- independent key/value storage types
- exact low-precision dense formats such as `F16`/`BF16`
- quantized formats such as `I8`, `Q4`, and future llama.cpp-style KV formats
- compressed committed-block formats such as `MSE_TURBOQUANT`

Example future config:

```json
{
  "kvBufferCache": {
    "kvKeyStorageType": "BF16",
    "kvValueStorageType": "Q8",
    "kvBlockStoragePolicy": "MSE_TURBOQUANT",
    "kvTurboQuantBits": 4
  }
}
```

Implementation notes:

- Keep mutable active blocks dense/writable unless a safe writable quantized format exists.
- Compress or quantize full committed blocks through `KvBlockStorage`.
- Preserve separate K/V policies because key error affects attention scores and value error affects output accumulation.
- Add provider/SIMD decode and dot/value primitives before making quantized KV a performance default.

## 2. Automatic Prefix Caching For KV2

Implement vLLM-style Automatic Prefix Caching using immutable KV2 blocks.

Target behavior:

- split prompt tokens into block-sized chunks
- hash full token blocks with model/runtime salt
- map matching prefix block chains to shared immutable `KvBlock` instances
- attach cached blocks by reference to new `KvCacheSession`s
- compute only uncached suffix/tail tokens
- maintain refcounts and eviction by bytes/blocks/recency

Desired structure:

```text
PrefixBlockCache
  token block hash chain -> shared KvBlock references

KvCacheSession
  attached committed prefix blocks
  mutable active tail block
```

Important invariants:

- only full committed blocks are shareable
- shared blocks must be immutable
- mutable tail blocks are never shared
- cache hits must preserve absolute positions and decode-start math
- layout/provider/dtype/model salt must match before attaching blocks

## 3. Disk-Based KV2

Add a disk-backed KV2 storage tier after in-memory immutable block sharing is stable.

Target use cases:

- long-running agents with large reusable contexts
- memory pressure relief for inactive shared prefix blocks
- optional persistence of expensive prompt prefixes

Possible design:

- `KvBlockStorage` gains disk-backed implementations or loadable handles
- `PrefixBlockCache` can evict cold blocks from memory to disk
- disk entries store block metadata, layout, dtype, token hash, and encoded bytes
- hot blocks are memory-resident; cold blocks hydrate on demand

Important constraints:

- disk I/O must not be hidden in attention hot paths without metrics
- loaded blocks must still satisfy immutable `KvBlock` semantics
- eviction/hydration must account for refcounts and active sessions
- disk format should support dense, quantized, and TurboQuant block layouts

## Priority

1. Configurable KV key/value precision.
2. KV2 Automatic Prefix Caching with in-memory immutable block sharing.
3. Disk-backed KV2 block storage and cold-block eviction.
