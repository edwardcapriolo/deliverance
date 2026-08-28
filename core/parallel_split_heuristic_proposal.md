# Parallel Split Heuristic Proposal

## Context

`parallelSplitSize` is currently derived from CPU count and provider multipliers, with optional fixed overrides. That is too blunt for model inference because the value also shapes native GEMM work chunks. The same split can be good for one model and catastrophic for another.

This proposal records the observed behavior and a deferred direction. It is not implemented yet.

## Benchmark Evidence

The prefill-only parameterized IT `QwenPrefillParallelSplitSweepIT` runs `batchForward(...)` for a fixed prompt and sweeps SIMD split sizes.

Observed on local Linux/arm64 test environment, prompt length 120:

| Model | Split | Prefill ms | Prefill tok/s |
|---|---:|---:|---:|
| Qwen3-0.6B-JQ4 | 4 | 3417.502 | 35.113 |
| Qwen3-0.6B-JQ4 | 8 | 2653.093 | 45.230 |
| Qwen3-0.6B-JQ4 | 16 | 2869.704 | 41.816 |
| Qwen3-0.6B-JQ4 | 32 | 2698.843 | 44.464 |
| Qwen3-0.6B-JQ4 | 64 | 2997.494 | 40.033 |
| Qwen3-4B-JQ4 | 4 | 16044.437 | 7.479 |
| Qwen3-4B-JQ4 | 8 | 14426.292 | 8.318 |
| Qwen3-4B-JQ4 | 16 | 14347.098 | 8.364 |
| Qwen3-4B-JQ4 | 32 | 13388.829 | 8.963 |
| Qwen3-4B-JQ4 | 64 | 87392.899 | 1.373 |

Additional observations:

- Split `2` is not a useful real candidate.
- Split `128` was bad for both 0.6B and 4B in testing.
- Split `64` is usable for 0.6B but catastrophic for 4B prefill.
- Split `32` is the best observed practical value for 4B prefill.

## Why A Global Multiplier Fails

The configured split affects the output-row chunk size used by projection kernels. A fixed split does not mean the same amount of work across models.

Approximate relative chunk footprint:

```text
0.6B: split 64 * width 1024 = 65,536 units
4B:   split 64 * width 2560 = 163,840 units
4B:   split 32 * width 2560 = 81,920 units
```

So `4B @ 32` is closer to `0.6B @ 64` than `4B @ 64` is. This points to cache/working-set pressure rather than CPU count alone.

## Ollama / vLLM Lesson

Ollama delegates this area mostly to llama.cpp. llama.cpp does not ask users to tune per-model native row chunk sizes. Instead it separates prompt/batch behavior from decode behavior with knobs like `n_batch`, `n_ubatch`, `n_threads`, and `n_threads_batch`.

vLLM is GPU-first and not solving this exact CPU SIMD chunking issue, but it also separates prefill and decode scheduling and uses model/hardware-aware defaults rather than asking users to tune per-model matmul chunks.

The useful lesson is not to copy a specific algorithm. The useful lesson is:

- Separate prefill and decode decisions.
- Use model and operation shape.
- Keep user-facing configuration simple.
- Avoid per-model manual tuning requirements.

## Proposed Direction

Replace provider-global multiplier-derived split selection with shape-aware defaults chosen from a small candidate set.

Candidate SIMD prefill splits:

```text
8, 16, 32, 64
```

Early heuristic direction:

```text
if embeddingLength >= 2048:
    prefer 32 for SIMD prefill
else:
    prefer 32 or 64 for SIMD prefill
```

A better version should consider operation shape:

```text
provider
phase: PREFILL or DECODE
M: batch/token count
K: input width
N: output rows
estimated bytes per output row
available cores
```

The split should be selected from a known-safe candidate set rather than computed as an arbitrary multiplier result.

## Open Questions

- Should prefill and decode have separate split policies immediately?
- Should SIMD and Panama maintain separate shape heuristics?
- Should the heuristic include a small startup microbenchmark, or only static model-shape rules?
- Should fixed overrides remain only as an escape hatch for benchmarking/debugging?
