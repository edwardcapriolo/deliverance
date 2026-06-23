# JQ4 Tensor Format

JQ4 is Deliverance's safetensors-friendly Q4 model layout. It is not GGUF. It keeps Hugging Face-style model directories and rewrites selected tensors into Deliverance Q4 tensors plus scale sidecars.

## Files

A quantized model directory usually contains normal model files:

```text
config.json
tokenizer.json
tokenizer_config.json
model.safetensors or model-00001-of-000NN.safetensors
model.safetensors.index.json
README.md
```

Deliverance quantized outputs also include:

```text
deliverance-quantization.json
.finished
```

The manifest records source/output paths, tensor dtype changes, generated sidecars, and shape transforms.

## Q4 Tensor Pair

A Q4 tensor is stored as two logical safetensor entries:

```text
model.layers.0.self_attn.q_proj.weight      # packed 4-bit values
model.layers.0.self_attn.q_proj.weight.qb   # per-block scale data
```

The main tensor stores packed 4-bit quantized values. The `.qb` sidecar stores block scale factors used to reconstruct approximate floating-point values during GEMM.

Conceptually:

```text
scale block | packed q4 data block
scale block | packed q4 data block
scale block | packed q4 data block
```

The sidecar form keeps the safetensors header simple and lets the weight loader recognize that a Q4 weight needs its matching `.qb` tensor.

## What Gets Quantized By Default

The default Q4 export policy quantizes the large attention and MLP projection matrices:

- `self_attn.q_proj.weight`
- `self_attn.k_proj.weight`
- `self_attn.v_proj.weight`
- `self_attn.o_proj.weight`
- `mlp.gate_proj.weight`
- `mlp.up_proj.weight`
- `mlp.down_proj.weight`

The default policy keeps embeddings, normalization weights, miscellaneous dense weights, and `lm_head` dense. Output-head Q4 can still be tested at load time with `withOutputHeadQuantization(DType.Q4)`.

## Why Not Quantize Everything

Not all tensors are equally valuable or safe to quantize.

- Large projection matrices dominate storage and memory bandwidth.
- Norm weights and small vectors are cheap to keep dense.
- Embeddings and output heads can affect quality and logits directly.
- Non-2D tensors are kept dense because Q4 export currently targets matrix-like tensors.

This policy is intentionally conservative: get most of the speed and size win from the largest matrices without making every tensor lossy.

## Runtime Counters

Benchmark profile counters show which dtype path was actually used:

```text
[profile-counter] sampler.output_projection.input_dtype.F32 count=256
[profile-counter] sampler.output_projection.weight_dtype.Q4 count=256
[profile-counter] mlpblock.down_quantize.input_dtype.F32 count=9216
```

Use these counters when comparing dense, QOD, JQ4, and output-head-Q4 runs.

## Related Pages

- [Quantize On Demand](quantize_on_demand.md)
- [Tensor engines and JQ4](tensor_engines_and_jq4.md)
- [Native SIMD kernels](native_simd_kernels.md)
