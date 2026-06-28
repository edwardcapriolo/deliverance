# Tensor Engines And JQ4

Deliverance spends most of its time multiplying model activations by model weights. JQ4 is Deliverance's way of keeping the model in a normal safetensors directory while shrinking the large weight matrices that dominate local inference.

The short version: smaller weights mean less memory traffic, and local CPU inference is often limited by memory bandwidth. When the same prompt and model architecture run with Q4 projection weights, token/sec can go up because the CPU has less weight data to move for every generated token.

For example, on a Qwen3-0.6B path we observed dense BF16 generation around the low teens in tokens/sec, while the QOD-generated `Qwen3-0.6B-JQ4` path reached about `24 tok/s` in a reasoning benchmark. Exact numbers depend on hardware, prompt length, native/Panama path, and output-head settings, but if we can get higher tokens/sec without a noticeable sacrifice in quality, we are doing our job well.

## The Bread-And-Butter Operation

Inside every transformer layer, Deliverance repeatedly does projection math like this:

```text
hidden_state x weight_matrix -> projected_state
```

In code this is usually one of the tensor operations:


```java
void batchDotProduct(AbstractTensor result, AbstractTensor a, AbstractTensor b, ...)
void dotProductBatchChunk(AbstractTensor[] result, AbstractTensor a, AbstractTensor[] b, ...)
```

In this sentence, "activation" means the live numbers flowing through the model for the current token or prompt. "Weight" means the learned matrix loaded from disk.

For Qwen3-4B, the hidden state width is `2560`. For one generated token, the activation going into an MLP projection is roughly a `1 x 2560` row of floats. In F32 that is only about `10 KB` of live data:

```text
1 token x 2560 floats x 4 bytes = 10,240 bytes
```

The MLP `up_proj` weight is much larger. Qwen3-4B has `intermediate_size=9728`, so that matrix is roughly `9728 x 2560` values:

```text
9728 x 2560 = 24,903,680 weights
```

Stored as BF16, that one matrix is about `47.5 MB`:

```text
24,903,680 weights x 2 bytes = 49,807,360 bytes
```

Stored as Q4, the packed values are about `11.9 MB` plus scale blocks:

```text
24,903,680 weights x 0.5 bytes = 12,451,840 bytes
```

So the live activation row is small, but the learned weight matrix is huge and is read over and over. JQ4 keeps the live math in a useful working dtype while shrinking the large repeated matrix.

That is why the important combinations are:

```text
BF16 x BF16 -> output
F32  x Q4   -> output
BF16 x Q4   -> output
```

That difference matters most on the big repeated matrices:

- attention query/key/value/output projections
- MLP gate/up/down projections
- optional output-head projection

For a Qwen3-style MLP `up_proj`, the dense weight matrix is large. In BF16 it costs two bytes per weight. In Q4 it is roughly half a byte per weight plus scale blocks. The arithmetic still has to happen, but the memory traffic for the weight matrix is much smaller.

## Why This Matters In A Real Turn

A single Qwen3-4B Q4 benchmark turn can run transformer blocks more than 9,000 times:

```text
[profile] transformerblock.forward              count=9216 total_ms=22829.248
[profile] causalselfattention.forward           count=9216 total_ms=10351.298
[profile] mlpblock.forward                      count=9216 total_ms=11218.508
[profile] sampler.output_projection             count=256  total_ms=4557.369
```

Those calls are dominated by matrix multiplies, attention score/value work, output projection, activation, and quantization boundaries. A single matrix optimization is not called once; it is called once per layer per token. That is why Q4 projection weights and native/Panama GEMM kernels can change visible tokens/sec.

## Tensor Engine Stack

Deliverance has three main tensor operation implementations:

- `NaiveTensorOperations`: straightforward Java loops and array-style access. Useful for correctness and fallback behavior.
- `PanamaTensorOperations`: Java Foreign Memory and Vector API operations using lane-wise hardware acceleration from Project Panama.
- `NativeSimdTensorOperations`: JNI/native C kernels for major GEMM paths, delegating operations not yet implemented natively back to the configured Java/Panama provider.

Startup logs tell you which provider is active:

```text
[main] INFO io.teknek.deliverance.model.ModelSupport - Machine Vector Spec: 128 Byte Order: LITTLE_ENDIAN
[main] INFO io.teknek.deliverance.model.AbstractModel - Tensor provider = Native SIMD Operations, parallelSplitSize = 32
[main] INFO io.teknek.deliverance.model.AbstractModel - Model type = Q4, Working memory type = F32, Quantized memory type = I8
```

`Machine Vector Spec` is what the JVM sees for vector lane width. Wider lanes let Panama process more values per vector instruction when the operation is implemented with the Vector API.

## Q4 And JQ4 Models

Deliverance Q4 model directories, often named with a `JQ4` suffix, store selected large projection weights in Deliverance's Q4 tensor format. Keeping the model in safetensors means normal model metadata, tokenizer files, and model cards can stay alongside quantized weights.

[Quantize On Demand](quantize_on_demand.md) can generate one of these directories locally:

```java
AutoModelForCausaLm.newBuilder(new ModelFetcher("Qwen", "Qwen3-0.6B"))
        .withQuantizeOnDemand(DType.Q4, "Qwen", "Qwen3-0.6B-JQ4")
        .buildLocalTransformerModel();
```

Then benchmark the generated model as a normal local model:

```text
edwardcapriolo/Qwen3-0.6B-JQ4
```

Example profile counters from a Qwen3-4B JQ4 run:

```text
[profile-counter] sampler.output_input_F32 count=256
[profile-counter] sampler.output_weight_Q4 count=256
[profile-counter] mlpblock.down_input_F32 count=9216
```

These counters show that hidden states are still computed in F32 while output-projection weights are Q4. That is the main local-inference tradeoff: keep compute stable enough for quality, but reduce bandwidth and storage for the large weight matrices.

## What It Looks Like In Practice

In a Qwen3-0.6B benchmark, the JQ4 model was the clear winner. The dense model was around the low-to-mid teens in tokens/sec, while the local JQ4 model hit about `24 tok/s` on the same style of reasoning workload:

```text
[deliverance] model=edwardcapriolo/Qwen3-0.6B-JQ4 case=builtin-reasoning-1 category=reasoning turn=2 prompt_tokens=405 generated=256 total_ms=13460.8 tok_s=24.62 finish=MAX_TOKENS
```

That is the sales pitch for JQ4: same model family, local safetensors, smaller projection weights, and a visible tokens/sec jump.

The profile also confirmed the hot output projection was using Q4 weights:

```text
[profile-counter] sampler.output_weight_Q4 count=256
```

The exact number will move with hardware, prompt length, and native/Panama path, but this is the kind of win Deliverance is trying to make easy: download a safetensors model, generate a JQ4 copy locally, then run it faster without leaving the Java stack.

## Related Pages

- [JQ4 tensor format](jq4_tensor_format.md)
- [Native SIMD kernels](native_simd_kernels.md)
- [Benchmarking](benchmarking.md)
