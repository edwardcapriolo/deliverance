# KVCache2 Prefix Snapshot Cache Plan

## Goal

Preserve the current prefix-cache feature for KVCache2 models without coupling KVCache2 to the old `KvBufferCache` classes.

This is intentionally not vLLM Automatic Prefix Caching. It is a temporary snapshot cache that keeps today's behavior while models migrate to KVCache2. True immutable shared-block APC can replace it later.

## Non-Goals

- Do not reuse `KvBufferCache.KvBuffer` as the storage payload.
- Do not add old/new bridge logic.
- Do not implement shared physical block reuse.
- Do not add refcounted APC yet.
- Do not add disk-backed KV2 in this pass.
- Do not make model-specific prefix-cache logic.

## Keep From Current Prefix Cache

Keep the existing simple data model concept:

```java
record CacheKey(Optional<String> salt, List<Integer> prefixTokens) {}
```

Keep these behaviors:

- block-aligned hits only
- same checkpoint interval policy
- longest matching prefix wins
- salt isolates entries
- max entries / LRU eviction
- "found hit, restore it, continue from prefix length"

Carry over these settings from `KvBufferCacheSettings`:

- `maxEntries`
- `blockSize`
- `maxPrefixTokensPerPrompt`
- `prefixCheckpointPolicy`
- `maxPrefixCheckpointsPerPrompt`
- `prefixCheckpointAnchors`

Compression can come later. Dense snapshots first.

## New Class

Add:

```text
core/src/main/java/io/teknek/deliverance/tensor/kv/KvPrefixSnapshotCache.java
```

Suggested public/package API:

```java
public final class KvPrefixSnapshotCache implements AutoCloseable {
    public record CacheKey(Optional<String> salt, List<Integer> prefixTokens) {}
    public record PrefixHit(int length) {}

    public PrefixHit lookupPrefix(int[] tokens, Optional<String> salt, KvCacheSession destination);
    public void storePrefix(int[] tokens, KvCacheSession source, Optional<String> salt);
    List<Integer> checkpointLengths(int tokenLength);
    public void close();
}
```

Internal entry shape:

```java
interface StoredPrefixEntry extends AutoCloseable {
    int length();
    void restoreTo(KvCacheSession destination);
}
```

Initial implementation:

```java
final class DenseStoredPrefixEntry implements StoredPrefixEntry
```

It stores copied key/value rows from a `KvCacheSession` and restores them into another `KvCacheSession`.

## Lookup Flow

```text
tokens + salt
-> checkpointLengths(min(tokens.length, maxPrefixTokensPerPrompt))
-> find longest CacheKey hit
-> restore stored rows into destination KvCacheSession
-> destination.advanceLength(prefixLength)
-> return PrefixHit(prefixLength)
```

No hit returns `null`.

The generation cursor then processes only the uncached suffix at original absolute positions, exactly like today.

## Store Flow

```text
tokens + source KvCacheSession
-> checkpointLengths(min(tokens.length, maxPrefixTokensPerPrompt))
-> for each prefix length:
     build snapshot completely outside map visibility
     insert if absent
     discard duplicate if another thread won
```

Only rows up to `prefixLen` are stored.

## Thread Safety Contract

Multiple users may infer concurrently.

Allowed:

- duplicate work
- two requests computing the same prefix at the same time
- one duplicate snapshot being discarded

Not allowed:

- corrupt cached entries
- partially written snapshots becoming visible
- eviction closing an entry while another thread is restoring it
- mutation of cached snapshots after insertion

Implementation rules:

1. Stored entries are immutable after construction.
2. Build snapshots fully before insertion.
3. Insert with `putIfAbsent`; close/discard losing duplicate entries.
4. Never expose stored tensors directly to callers.
5. Restore copies from stored entry into caller's `KvCacheSession`.
6. Synchronize lookup/restore/eviction over the cache map lock.
7. Synchronize `close()` over the same cache map lock.

Simple map shape:

```java
private final Map<CacheKey, StoredPrefixEntry> prefixCache =
    Collections.synchronizedMap(new LinkedHashMap<>(16, 0.75f, true) {
        protected boolean removeEldestEntry(Map.Entry<CacheKey, StoredPrefixEntry> eldest) {
            ... close eldest under map lock ...
        }
    });
```

Lookup should restore under synchronization:

```java
synchronized (prefixCache) {
    StoredPrefixEntry best = ...;
    if (best != null) {
        best.restoreTo(destination);
        return new PrefixHit(best.length());
    }
}
```

This serializes prefix-cache restores, but it is simple and safe. It is acceptable for a temporary snapshot cache.

## Integration

Add a `KvPrefixSnapshotCache` field to `AbstractModel` beside `KvCacheManager`.

Wire only KVCache2 local generation:

```java
LocalKvCache2GenerationSession
```

Constructor:

```java
this.effectiveCacheSalt = withActiveAdapterScope(parameters.cacheSalt);
this.kvSession = model.newKvCacheSession();
PrefixHit hit = model.kvPrefixSnapshotCache.lookupPrefix(promptTokens, effectiveCacheSalt, kvSession);
this.prefixLength = hit == null ? 0 : hit.length();
```

Prefill:

```java
if (cursor.hasTokensToProcess()) {
    last = model.batchForward(cursor.tokensToProcess(), cursor.startPosition(), kvSession);
    model.kvPrefixSnapshotCache.storePrefix(promptTokens, kvSession, effectiveCacheSalt);
} else {
    last = model.forward(cursor.replayToken(), cursor.replayPosition(), kvSession);
}
```

Keep `GenerationCursor` unchanged.

## Metrics

Use new KVCache2 names:

- `kvcache.v2.prefix.lookup`
- `kvcache.v2.prefix.hits`
- `kvcache.v2.prefix.misses`
- `kvcache.v2.prefix.store`
- `kvcache.v2.prefix.evict`
- `kvcache.v2.prefix.restore`
- `kvcache.v2.prefix.snapshot.bytes`

## Tests

Add:

```text
core/src/test/java/io/teknek/deliverance/tensor/kv/KvPrefixSnapshotCacheTest.java
```

Unit tests:

- exact block hit
- longest matching prefix wins
- no hit below block size
- salt isolates entries
- eviction removes old entry
- restored rows match source rows
- restored prefix advances session length
- suffix prefill can append after restored prefix
- duplicate concurrent stores do not corrupt entries
- concurrent lookup/store does not expose partially built entries

Generation-level test:

- Qwen tiny first request stores prefix
- second request with same prompt gets nonzero prefix length
- decode starts after full prompt length
- no old `KvBufferCache.KvBuffer` involved

## Deletion Path

When true KVCache2 APC exists:

1. Replace `KvPrefixSnapshotCache` lookup/store with APC lookup/store.
2. Keep `GenerationCursor` behavior.
3. Delete `KvPrefixSnapshotCache` if redundant.
4. Delete old `KvBufferCache` prefix snapshot code after all migrated models use KVCache2.
