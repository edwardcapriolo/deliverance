# Phase 2: File, Config, And Weight Inventory

## Status

Remote file inventory complete. Local Java tensor inventory pending because the checkpoint is not cached and is approximately 17.6 GB.

## Repository Snapshot

- Repository: `ibm-granite/granite-4.1-8b-llmon-preview`
- Revision: `e1787b0329a1abf7e58337a9988a68884a5ec629`
- Pipeline: `text-generation`
- Library: `transformers`
- Base model: `ibm-granite/granite-4.1-8b-base`
- Safetensors parameter summary: `8,791,592,960` BF16 parameters

## Files

| File | Reported size | Purpose |
| --- | ---: | --- |
| `.gitattributes` | 1.52 KB | Repository attributes |
| `README.md` | 23.3 KB | Model card and usage |
| `chat_template.jinja` | 14.2 KB | LLMON prompt renderer |
| `config.json` | 891 B | Model configuration |
| `generation_config.json` | 173 B | Generation defaults |
| `model.safetensors` | 17.6 GB | Single BF16 checkpoint |
| `tokenizer.json` | 7.15 MB | Tokenizer vocabulary and tokenizer state |
| `tokenizer_config.json` | 398 B | Tokenizer configuration |

There is no safetensors index file because the model uses one checkpoint file.

## Existing Java Tooling

`safetensors/src/main/java/io/teknek/reckoning/safetensors/ModelInspectionTool.java` already provides Java inspection records and quantization classification.

It currently enumerates `TensorInfo` metadata and reports:

- Tensor names.
- Dtypes and shapes.
- Estimated bytes.
- Logical split and split-part status.
- Default Q4 selection.
- Q4 eligibility and quantization notes.

`SafetensorsInspector` provides a Java command-line metadata listing.

The required model-specific inventory still needs a Java run against the downloaded checkpoint that loads each tensor through the real `DefaultWeightLoader` path and emits a Markdown report.

## Known Inventory Facts

- Stored model dtype is BF16.
- The model is approximately 8B parameters; the Hugging Face page reports approximately 9B parameters.
- The checkpoint is not sharded.
- The model includes a 14.2 KB custom chat template with random two-character LLMON block IDs.
- The tokenizer and chat template are part of the model behavior and must be inventoried together with weights.

## Blocking Decision

The exact tensor names, ranks, shapes, and quantization candidates cannot be reported from `config.json` or the model-card API alone.

The Java inventory requires downloading the 17.6 GB checkpoint and running the loader. No Python or shell tensor inspection is being used.

## Next Step After Approval

Run or extend the Java `ModelInspectionTool` to:

1. Locate/download this model with `ModelFetcher`.
2. Enumerate every tensor from the safetensors header.
3. Load each tensor sequentially through `DefaultWeightLoader`.
4. Report actual shapes, ranks, dtypes, byte sizes, sidecars, and quantization eligibility.
5. Write `model-inventory.md` under this dossier.
