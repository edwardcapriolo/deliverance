# Lighter And Tensor2

`Lighter` is Deliverance's newer tensor operation layer. It is intentionally smaller and stricter than the older `TensorOperations` interface.

The design goal is simple: make hot operations explicit, prove them with shared tests, and let each provider either support the operation or reject it. A provider should not quietly fall into a slow path that makes a benchmark look correct while hiding why production is slow.

## Why Not The Old Delegate Style

The older tensor provider stack made it easy to write defensive code:

```text
try native
if unsupported, delegate to another provider
if still unsupported, fall back again
```

That was useful while building model support, but it had a cost:

- Easy to fall back.
- Hard to know which provider actually ran.
- Hard to optimize because correctness could come from a slow fallback.
- Hard to reason about performance when a "native" path silently delegated to Panama or naive Java.

That pattern is safe for early bring-up, but it is bad for performance work. It lets us accidentally ship code that works only because a slower provider rescued it.

## The Tensor2 Contract

`tensor2.TensorOps` uses an explicit result:

```java
Either<OpSupport, Void>
```

A provider does one of two things:

- Returns `Right(null)` after it performs the operation.
- Returns `Left(OpSupport.Unsupported)` when it does not support the operation.

`Lighter` owns provider dispatch:

```java
Lighter lighter = new Lighter(metricRegistry, Map.of(
    TensorProviderKind.SIMD, nativeOps,
    TensorProviderKind.PANAMA, new PanamaOps(),
    TensorProviderKind.NAIVE, new NaiveOps()
));
```

The provider order is explicit. The current tests assert that `SIMD` is tried before `PANAMA` for `batchDotProduct`.

`Lighter` itself does not know about native code. Native support is injected from the model builder when the native module is present.

## Native Wiring Boundary

The native tensor2 provider lives in the `native` module:

```text
native/src/main/java/io/teknek/deliverance/tensor2/NativeOps.java
```

The tensor module does not import it and does not reflectively load it.

`AutoModelForCausaLm` is allowed to do optional native discovery because it already owns model runtime construction. It reflects for `NativeOps`, asks `NativeOps.isAvailable()`, and injects the provider into the model's `Lighter`.

That keeps the boundary clear:

- `tensor`: interfaces, Panama, naive, operation objects.
- `native`: native tensor2 provider and generated jextract bindings.
- `core`: model builder wires optional native support into a model.

## BatchDotProduct

The first tensor2 operation added for prefill attention is `BatchDotProduct`.

It computes a block of dot products:

```text
result[resultRow, resultColumn] = dot(a[aRow], b[bRow])
```

with explicit row and column windows:

```java
new BatchDotProduct()
    .result(result)
    .a(query)
    .b(keys)
    .aRowOffset(queryRowStart)
    .aColumnOffset(queryOffset)
    .bColumnOffset(kvOffset)
    .columnLength(headSize)
    .resultRowOffset(0)
    .bRowOffset(0)
    .rowChunkSize(visibleRows);
```

The important field is `aRowOffset`. It lets attention multiply a row window from `query` without copying that row window into a temporary tensor.

Before this, tiled attention needed a copied `queryBlock` just because the older batch-dot method could not start reading from a non-zero input row. That was correct, but it created large direct-buffer churn in long prompts.

## Tensor2 Native SIMD

The tensor2 native C entrypoint is separate from the old SIMD API:

```text
native/src/main/c/tensor2/tensor2_native.h
native/src/main/c/tensor2/tensor2_native.c
```

The exported function is:

```c
tensor2_status tensor2_batch_dot_f32_f32(...)
```

It is not a wrapper around the old `gemm_f32` symbol. The implementation is copied/adapted into the tensor2 native file under the tensor2 API.

The current implementation supports `F32 x F32` and uses SIMD:

- NEON on ARM/AArch64.
- AVX512 when available.
- AVX/FMA on x86 otherwise.
- Scalar only for final tail elements.

The Java binding is generated with jextract:

```text
native/jextract_tensor2_native.sh
```

The generated Java files live under:

```text
native/src/main/java/io/teknek/deliverance/tensor/operations/tensor2native/
```

## Shared Fuzz Tests

Batch-dot has the same test style as tensor2 multiply-accumulate.

The shared case generator is:

```text
native/src/test/java/io/teknek/deliverance/tensor2/BatchDotProductFuzzCases.java
```

The fuzzer is:

```text
native/src/test/java/io/teknek/deliverance/tensor2/Tensor2LighterBatchDotProductFuzzTest.java
```

It compares providers through `Lighter` against a naive `Lighter` reference.

Current coverage includes:

- input row offsets
- input/output column offsets
- result row offsets
- B row offsets
- row chunks
- odd sizes and tails
- fixed cases and randomized fuzz cases

The tensor2 native path also has a focused native parity test:

```text
native/src/test/java/io/teknek/deliverance/tensor2/NativeOpsBatchDotProductTest.java
```

## Shot At The Title

Qwen prefill had a very practical problem: even a smallish web page could be slow to prefill. The model could decode reasonably, but prompt ingestion made local agent workflows painful.

The direct prefill benchmark made the problem visible without tokenizer, chat template, sampling, or decode noise:

```text
benchmarks/run-prefill-token-smoke-benchmark.sh
benchmarks/run-prefill-token-deep-benchmark.sh
```

The smoke benchmark uses deterministic token IDs and direct `batchForward(...)` calls. That showed Qwen prefill was attention-heavy and CPU utilization was too low.

The important attention math is not complicated:

```text
scores = Q_block * K_visible^T
weights = softmax(scores)
output = weights * V_visible
```

The old shape did too much row-by-row work around the kernel. The improved shape batches the score multiply by query block and head, and then uses tensor2 `BatchDotProduct` so the query tile is read by row offset instead of copied.

For Qwen3-4B-JQ4, the clean smoke benchmark moved approximately from:

```text
500-token prefill: ~17.75 tok/s
```

to:

```text
500-token prefill: ~37 tok/s
```

For Qwen3-0.6B-JQ4, the 500-token case moved approximately from:

```text
~57 tok/s
```

to:

```text
~173 tok/s
```

That is not the end of prefill work. At longer contexts, causal attention still gets more expensive because every later chunk sees a longer prefix. But the work is now much healthier:

- CPU utilization is much higher.
- The hot multiply is a clean tensor2 operation.
- The native provider is explicit and tested.
- Long runs avoid the worst direct-buffer churn from copied query blocks.

Combined with KV-cache work, this is the path toward making local code-agent prompts feel usable instead of waiting on prefill for every moderately sized context.

## What Comes Next

The next tensor2 work should follow the same pattern:

1. Identify one hot primitive with a clean benchmark.
2. Add a first-class tensor2 operation object.
3. Implement Naive and Panama providers.
4. Add shared fuzz cases.
5. Add NativeOps support in the native module.
6. Inject native support explicitly from model construction.
7. Keep old `TensorOperations` out of new semantics unless a legacy caller needs it.

Likely next dtype/provider additions:

- `F32 x I8`
- `F32 x Q4`
- `I8 x Q4`
- reusable scratch for attention score buffers
- a tensor2 softmax/value-accumulation primitive
