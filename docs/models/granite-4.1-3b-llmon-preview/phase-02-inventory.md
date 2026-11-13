# Phase 2: File, Config, And Weight Inventory

## Status

Complete for operator review. The inventory was produced by Java using Reckoning's `DefaultWeightLoader` and loaded every tensor sequentially.

## Java Tooling

The inventory uses:

```text
safetensors/src/main/java/io/teknek/reckoning/safetensors/ModelInspectionMain.java
safetensors/src/main/java/io/teknek/reckoning/safetensors/ModelInspectionTool.java
```

The tool now loads every tensor through `DefaultWeightLoader.loadRef(...)` before reporting it and writes a Markdown report with repository file sizes and tensor details.

## Model Files

Cached model directory:

```text
/home/ecapriolo/.reckoning/ibm-granite_granite-4.1-3b-llmon-preview
```

| File | Bytes |
| --- | ---: |
| `.finished` | 0 |
| `README.md` | 22864 |
| `chat_template.jinja` | 14200 |
| `config.json` | 889 |
| `generation_config.json` | 173 |
| `model.safetensors` | 7319517272 |
| `tokenizer.json` | 7153752 |
| `tokenizer_config.json` | 398 |

## Tensor Summary

- Tensor count: `363`
- Stored dtype: `BF16` for all tensors.
- Logical split parents: `0`.
- Split parts: `0`.
- Current default Q4 selection: `280` tensors.
- Current Q4 eligibility: `282` tensors.
- Observed weight structures are standard 1D normalization tensors and 2D projection/embedding tensors.
- No unexpected 7D or other high-rank tensors were found.

The complete tensor-by-tensor report is:

```text
docs/models/granite-4.1-3b-llmon-preview/model-inventory.md
```

## Important Findings

- The checkpoint is a single safetensors file rather than a shard set.
- The checkpoint contains no existing quantization sidecars.
- Most projection weights are eligible for the current Q4 quantizer based on shape and filter rules.
- Embedding and normalization tensors are excluded by the current default quantizer policy where appropriate.
- The `chat_template.jinja` file is a first-class model artifact and is larger than the JSON config; it must be preserved and reproduced in the prompt path.

## Lighter And Kernel Cross-Check

The inventory shapes fit the current TensorRef provider contracts.

- Model weights are BF16 and the default working memory is F32.
- `Lighter.batchDotProduct` supports F32 activations with BF16 weights and F32 output.
- Panama has an `F32 x BF16 -> F32` batch GEMM path.
- Native Tensor2 has an `F32 x BF16 -> F32` path.
- Q4 conversion is shape-compatible: the projection inner dimensions `2560` and `8192` are multiples of the Q4 block size `32`.
- `Lighter` and Panama support F32 activations with Q4 weights when the Q4 scale sidecar is present and ranges are aligned.
- Q/K/V shapes are compatible with the model dimensions: query projection width `2560`, KV projection width `512`, and head size `64`.
- GQA is shape-compatible: `40` query heads divided by `8` KV heads gives a head group size of `5`.
- The `100352 x 2560` output head is currently excluded by the default Q4 filter because `lm_head` weights are kept dense.

No kernel shape blocker is visible for the dense BF16 checkpoint or the current Q4 projection path.

This does not yet prove the Granite block semantics, scaling multipliers, RoPE behavior, or LLMON prompt behavior.

## Command

The report was generated with the Java main:

```text
mvn -pl safetensors org.codehaus.mojo:exec-maven-plugin:3.5.0:java \
  -Dexec.mainClass=io.teknek.reckoning.safetensors.ModelInspectionMain \
  -Dexec.args="inspect ibm-granite granite-4.1-3b-llmon-preview model-inventory.md markdown"
```

## Next Phase After Approval

Inspect the tokenizer and chat template behavior, then inspect the exact Transformers implementation and model-specific tests.
