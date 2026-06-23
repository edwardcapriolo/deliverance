# Quantize On Demand

Quantize On Demand, or Q.O.D., lets Deliverance generate a local quantized copy of a model the first time you ask for it, then reuse that generated model on later runs.

The common flow is:

1. Resolve the source model from the Deliverance cache, downloading it if allowed.
2. Check whether the requested quantized target already exists locally.
3. If the target exists and is complete, load it directly.
4. If the target is missing, quantize the source into a staging directory.
5. Move the staged model into `~/.deliverance/<owner>_<model>`.
6. Load the generated target model.

## Builder Usage

```java
ModelFetcher fetch = new ModelFetcher("Qwen", "Qwen3-0.6B");

try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
        .withQuantizeOnDemand(DType.Q4, "Qwen", "Qwen3-0.6B-JQ4")
        .buildLocalTransformerModel()) {
    // Generate as usual.
}
```

The target owner/model names control the generated cache directory. The example above writes to:

```text
~/.deliverance/Qwen_Qwen3-0.6B-JQ4
```

## Offline And Local-Only Usage

Q.O.D. respects the builder download policy for the source model:

```java
AutoModelForCausaLm.newBuilder(new ModelFetcher("Qwen", "Qwen3-0.6B"))
        .withDownload(false)
        .withQuantizeOnDemand(DType.Q4, "Qwen", "Qwen3-0.6B-JQ4")
        .buildLocalTransformerModel();
```

With downloads disabled, the source model must already exist in the local Deliverance cache. Generated quantized targets are always loaded locally and do not contact Hugging Face.

If the target directory already exists but is incomplete, Deliverance fails rather than overwriting it. Remove the incomplete target directory manually if you want to regenerate it.

## What Gets Quantized

The default Q4 policy quantizes large attention and MLP projection matrices:

- `self_attn.q_proj.weight`
- `self_attn.k_proj.weight`
- `self_attn.v_proj.weight`
- `self_attn.o_proj.weight`
- `mlp.gate_proj.weight`
- `mlp.up_proj.weight`
- `mlp.down_proj.weight`

The default policy keeps embeddings, normalization weights, miscellaneous dense tensors, and `lm_head` dense. If you want to test output-head Q4 without changing the persisted model directory, use the load-time output-head option:

```java
AutoModelForCausaLm.newBuilder(fetch)
        .withQuantizeOnDemand(DType.Q4, "Qwen", "Qwen3-0.6B-JQ4")
        .withOutputHeadQuantization(DType.Q4)
        .buildLocalTransformerModel();
```

## Generated Files

Q.O.D. generated model directories include the normal copied model metadata and rewritten safetensor weights. They also include Deliverance-specific provenance files:

- `README.md`: prepends a Deliverance Q.O.D. summary, original vs quantized local size, and a note that the copied model-card content belongs to the original model authors.
- `deliverance-quantization.json`: records source/output directories, target dtype, size information, tensor dtype changes, generated `.qb` sidecars, and shape-normalization transforms.
- `.finished`: marks the local generated model as complete for offline loading.

## Progress Logging

Q.O.D. emits INFO logs during long conversions. Core tests include `simplelogger.properties`, so these logs appear in normal Maven and IDE test output.

Typical logs include:

```text
Using existing quantized model target /Users/.../.deliverance/Qwen_Qwen3-4B-JQ4
Quantized model target ... is missing; resolving source ...
Creating quantized model target ... via staging directory ...
Quantizing model from ... to ... with target dtype Q4
Loaded 196 tensors for quantization from ...
Quantizing tensor 42/196 model.layers.5.mlp.up_proj.weight from BF16 to Q4
Quantization progress: 50/196 tensors processed in 8 seconds
Writing quantized model weights to ...
Finished quantizing model to ... in 30 seconds
```

## Practical Notes

- Quantization can take seconds to minutes depending on model size and disk speed.
- The first run pays the conversion cost; later runs reuse the generated cache target.
- Q4 usually improves local CPU throughput and reduces disk footprint, but it can change sampled text.
- Exact golden text is brittle with sampled Q4 generation; semantic assertions are usually better for integration tests.
- Native artifacts are platform-specific, but generated model directories are safetensors plus metadata and are not tied to a native build output directory.
