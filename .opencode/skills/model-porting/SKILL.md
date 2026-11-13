---
name: model-porting
description: Use for interactive, faithful ports of Hugging Face, Transformers, or Sentence Transformers models into Reckoning; stop after every phase for operator approval.
---

# Model Porting

Use this skill for a model-family port into Reckoning. This is an interactive workflow, not a one-shot implementation task.

## Required Skills

Use the `faithful-port` skill for source comparison, implementation fidelity, tensor checkpoints, and parity claims.

Use the `dead-reckoning` skill for deterministic source/target traces, golden files, runtime profiles, and regression comparison.

Read both skills before beginning implementation work.

## Interactive Rule

Work on one phase at a time.

At the end of every phase:

1. Update the phase document under `docs/models/<model>/`.
2. Summarize completed work, evidence, unknowns, risks, and proposed next steps in chat.
3. Stop and wait for explicit operator approval.

Do not advance phases automatically. Discuss unforeseen architecture, data, performance, or tooling problems with the operator before continuing.

## Living Documentation

Create a model dossier at:

```text
docs/models/<model-slug>/
```

Maintain living documents such as:

```text
README.md
phase-01-documentation.md
phase-02-inventory.md
phase-03-source-tests.md
phase-04-characterization.md
phase-05-port.md
phase-06-validation.md
phase-07-performance.md
phase-08-quantization.md
references/
```

Record source URLs, revisions, retrieval dates, decisions, rejected alternatives, test commands, results, and remaining risks.

## Phase 1: Documentation And Model Selection

The first file to inspect is the selected model's `config.json`. Do this before reading implementation code or writing target code.

Create `docs/models/<model-slug>/config-field-analysis.md` and itemize every configuration parameter.

For each parameter, record:

```text
name
JSON type
value
known Reckoning equivalent
new to Reckoning
currently ignored
required runtime behavior
source documentation/code
status: understood | needs investigation | blocking
```

Stop after this inventory if any field may affect model execution but its semantics are not understood.

Then select and document the exact checkpoint before implementation.

- Read the model card and linked official material.
- Inspect the selected `config.json`, tokenizer files, generation configuration, and checkpoint metadata.
- Read the relevant Transformers implementation and tests.
- Read Sentence Transformers sources and tests when the model has an embedding or SentenceTransformer wrapper.
- Search Reckoning for existing architecture and model-family support.
- Prefer instruction-tuned or chat-tuned checkpoints.
- Do not select a base checkpoint without recording the reason.

Document whether the model is supported, partially supported, or a new architecture, and identify what is unique in its forward pass.
Explicitly summarize every configuration field that is new, ignored, or not yet understood.

## Phase 2: File, Config, And Weight Inventory

Build a Java inventory utility or test using Reckoning loaders. Do not use Python or shell parsing unless Java cannot reasonably inspect the artifact.

Inventory:

- Every repository file and size.
- Config, tokenizer, chat-template, generation, and checkpoint-index files.
- Every checkpoint shard and total size.
- Every tensor name, shape, rank, dtype, quantization sidecar, and layout.
- Tied/shared weights and unusual tensor suffixes.
- Unknown fields and unsupported dtypes.

Create a typed Reckoning config class and a test covering every encountered field and value type.

For each unknown or ignored field, document its upstream semantics and whether it is blocking, safe to ignore, or not applicable.

Do not invent scaled-down tensor names or shapes without documenting that a fixture is synthetic.

## Phase 3: Source And Test Analysis

Create a source-test port matrix from Transformers and applicable Sentence Transformers tests.

For each relevant test, record:

```text
source test
source assertion
target test
fixture/input
expected checkpoint or output
status
```

Classify tests as framework, model-family, model-specific, tokenizer, generation/cache, embedding/pooling, checkpoint-loading, or quantization tests.

Use the `faithful-port` rules. Preserve source branches, operation order, cache behavior, and dtype transitions. Do not use generated prose as the primary oracle.

## Phase 4: Dead Reckoning Baseline

Adapt Dead Reckoning to download and run the exact selected model using the standard prompt and deterministic settings.

Capture:

- Model and source revision.
- Rendered prompt and input token IDs.
- Selected tokens and response.
- Load, prefill, decode, and total timings.
- Runtime/provider metadata.
- Intermediate checkpoints when architecture analysis requires them.

The baseline is source evidence. It is not a substitute for tensor-level parity.

## Phase 5: Characterization Tests

For every genuinely new component, add a characterization test before full integration.

Use as much production code as possible with a small deterministic checkpoint fixture using realistic shapes and dtypes.

Examples:

- New attention mechanism.
- Mixture-of-experts routing.
- New RoPE variant.
- Recurrent or state-space cache.
- Specialized normalization.

Use an independent small reference where practical. Assert shapes, masks, routing, finite values, ownership, and output behavior.

Production code must use provider-backed SIMD/Panama/native operations. Scalar code is allowed only as a bounded test oracle.

## Phase 6: Model Port And Validation

Build the model root and transformer blocks only after the configuration and characterization work is understood.

- Map exact checkpoint names.
- Apply the same transposes and layouts as Transformers.
- Preserve dtypes and quantization sidecars.
- Add warning-level progress logs at major load and forward boundaries.
- Add `InferenceProfiler` timers and counters for model load, blocks, projections, attention, MLP, cache updates, and logits.
- Ensure slow or stuck stages identify the model, layer, stage, shape, dtype, provider, and elapsed time.

Add a Dead Reckoning integration entry for the model. Add a profiler showcase IT using the existing profiler benchmark base where applicable.

The normal correctness IT must compare against an immutable golden. Never compare against the newest generated trace and never replace a golden automatically.

Use environment-specific golden paths such as:

```text
arm64-Darwin/<test>.jsonl
aarch64-Linux/<test>.jsonl
```

Generated actual traces belong under `target/` and must not be committed.

## Phase 7: Performance Testing

Build a model-specific benchmark using Java and existing Reckoning benchmark infrastructure.

Capture at least one clean baseline run and record:

- Hardware and OS.
- JDK and native provider.
- Model revision and dtype.
- Load time.
- Prefill throughput.
- Decode throughput.
- Latency and memory.
- Profiler timers and counters.

Compare against Reckoning models of similar parameter count, quantization, and workload. Record results in the model dossier.

## Phase 8: Quantization

Inventory quantization candidates and exclusions.

- Identify weights that can be quantized safely.
- Identify weights that must remain full precision.
- Document dtype, block size, scale, and layout requirements.
- Add `ModelQuantizer` rules when required.
- Build the quantized checkpoint.
- Add IT and parity coverage against the dense model.
- Compare selected tokens, response structure, tensor health, numerical drift, and performance.

Do not claim quantized support until dense-versus-quantized behavior is sufficiently characterized.

## Completion Gate

Do not call the port complete until:

- The selected checkpoint and source revision are documented.
- Config and weight inventory is complete.
- New components have characterization tests.
- Relevant source tests have a target mapping.
- Dead Reckoning source and target traces are captured.
- Immutable environment-specific goldens pass.
- Profiler and performance evidence exists.
- Quantization behavior is documented when applicable.
- Unsupported branches and remaining parity risks are explicit.
