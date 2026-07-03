# Prefix Cache MSE TurboQuant

Deliverance has an experimental prefix-cache storage mode named `MSE_TURBOQUANT`.

This feature compresses immutable prefix-cache snapshots. It does not change the live KV cache format and does not change attention kernels.

## What It Does

During prefix-cache storage, Deliverance normally copies KV rows into a full snapshot. Those snapshots are exact but large.

`MSE_TURBOQUANT` stores the same prefix snapshot as quantized row vectors:

- normalize each KV row by its L2 norm
- apply a deterministic sign-Hadamard rotation
- quantize rotated coordinates with a Lloyd-Max normal codebook
- pack codebook indices into bytes
- store per-row norms

On a prefix-cache hit, Deliverance decodes the quantized rows back into a normal F32/BF16 `KvBuffer`, then copies that buffer into the live request KV cache.

## What It Is Not

This is not full Google TurboQuant from the paper.

The paper's inner-product TurboQuant also applies a QJL residual correction for unbiased inner-product estimation. Deliverance's current feature is the MSE reconstruction part only. Because it decodes back into normal KV rows before attention, it is best described as MSE TurboQuant-style prefix snapshot storage.

## Why It Exists

Generic byte compression did not work on real KV snapshots. LZ4 made the payload slightly larger in a Qwen3-4B benchmark because dense F32 KV bytes were effectively incompressible.

MSE TurboQuant produced real size savings. In one Qwen3-4B-JQ4 benchmark case:

- raw prefix bytes: `94,371,840`
- encoded bytes: `11,888,640`
- size ratio: about `12.6%`
- memory reduction: about `87%`
- effective capacity gain: about `8x`

That makes much longer prefix-cache windows practical.

## Tradeoffs

Benefits:

- Much smaller prefix-cache entries.
- Lower allocator/cache pressure.
- Larger `maxPrefixTokensPerPrompt` becomes feasible.
- More checkpoints can be retained without exploding memory.

Costs:

- Prefix storage is slower because rows must be encoded.
- Prefix hits require decode into a temporary KV buffer.
- Restored KV values are approximate, not exact.
- Generated output may drift compared with exact raw prefix cache.
- This is not a live KV-cache memory reduction. Decode-time attention still uses normal KV pages.

## Configuration

For JSON model configs:

```json
{
  "kvBufferCache": {
    "maxPrefixTokensPerPrompt": 8192,
    "prefixCheckpointPolicy": "START_AND_END",
    "maxPrefixCheckpointsPerPrompt": 12,
    "prefixCompression": "MSE_TURBOQUANT",
    "prefixTurboQuantBits": 4
  }
}
```

For the Qwen nanocode web server, the same settings are exposed through Spring properties:

```properties
deliverance.kv.prefix.max-tokens=8192
deliverance.kv.prefix.checkpoint-policy=START_AND_END
deliverance.kv.prefix.max-checkpoints=12
deliverance.kv.prefix.compression=MSE_TURBOQUANT
deliverance.kv.prefix.turboquant.bits=4
```

## Metrics

When `--profile-stages` is enabled, benchmarks can show:

- `kvbuffercache.prefix.store`
- `kvbuffercache.prefix.turboquant.encode`
- `kvbuffercache.prefix.turboquant.decode`
- `kvbuffercache.prefix.turboquant.raw.bytes`
- `kvbuffercache.prefix.turboquant.encoded.bytes`

Useful derived value:

```text
encoded_bytes / raw_bytes
```

Lower is better for memory. Store/decode timings show the latency cost.

## Current Recommended Use

Use `MSE_TURBOQUANT` for experiments where prefix-cache capacity matters more than exact KV reconstruction.

Good candidates:

- long system prompts
- coding-agent sessions
- repeated prompts with large stable prefixes
- memory-constrained prefix-cache experiments

Avoid treating it as a default quality-preserving cache until model-specific output drift is measured.
