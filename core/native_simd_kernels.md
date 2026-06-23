# Native SIMD Kernels

Deliverance can run tensor operations through Java/Panama or through JNI-backed native SIMD kernels. The native provider is selected automatically when the `deliverance` native library is available on `java.library.path` or loadable from the jar.

Startup logs identify the active provider:

```text
[main] INFO io.teknek.deliverance.model.AbstractModel - Tensor provider = Native SIMD Operations, parallelSplitSize = 32
```

## What A GEMM Kernel Does

GEMM means general matrix multiplication. In Deliverance, GEMM-style kernels power most projection work:

```text
output = activation_or_hidden_state x weight_matrix
```

These kernels run for attention projections, MLP projections, and output projection. The native provider specializes common dtype combinations so quantized weights can be multiplied without first expanding the full matrix to F32.

## Native Batch Dot Product Paths

`NativeSimdTensorOperations.batchDotProduct(...)` routes these input/weight combinations to native code:

- BF16 x BF16
- BF16 x Q4
- F32 x F32
- F32 x BF16
- F32 x Q4
- I8 x Q4

The batched variant, `dotProductBatchChunk(...)`, supports the same broad combinations for multiple result/weight tensors:

- BF16 x BF16 batch
- BF16 x Q4 batch
- F32 x F32 batch
- F32 x BF16 batch
- F32 x Q4 batch
- I8 x Q4 batch

These paths are important for models like Gemma, Llama, Mistral, Mixtral, and Qwen where the same projection shape repeats layer after layer.

## What Still Delegates To Panama

Native SIMD is not a complete replacement for the Java tensor engine. Some operations still delegate to the configured Java/Panama provider. One important example today is `saxpy(...)`, used during attention value accumulation:

```java
NativeSimdTensorOperations.saxpy(...) -> delegate.saxpy(...)
```

So a run can report `Native SIMD Operations` while attention value accumulation still uses the Panama implementation. This is why stage profiling matters: a single provider name does not mean every hot operation is native C.

## Profiling Example

In a Qwen3-4B JQ4 benchmark, attention score/value and MLP projection dominate large parts of the request:

```text
[profile] causalselfattention.score_value       count=13788 total_ms=3602.329
[profile] causalselfattention.qkv_projection    count=9216  total_ms=3543.792
[profile] mlpblock.gate_up_projection           count=9216  total_ms=5456.746
[profile] sampler.output_projection             count=256   total_ms=4557.369
```

This tells us where native work matters next. If `batchDotProduct` dominates, GEMM specialization is the right path. If `saxpy` or softmax dominates inside `score_value`, the next native kernel may not be a GEMM at all.

## Native Build Layout

Native artifacts are platform-specific and are written under classifier-specific directories:

```text
native/target/native-lib-only/osx-aarch_64
native/target/native-lib-only/linux-aarch_64
native/target/native-lib-only/linux-x86_64
```

The benchmark scripts detect the classifier and pass the correct library path. You can override detection with:

```sh
DELIVERANCE_NATIVE_CLASSIFIER=osx-aarch_64 ./run-qwen-single-benchmark.sh
```

## Related Pages

- [Tensor engines and JQ4](tensor_engines_and_jq4.md)
- [JQ4 tensor format](jq4_tensor_format.md)
- [Benchmarking](benchmarking.md)
