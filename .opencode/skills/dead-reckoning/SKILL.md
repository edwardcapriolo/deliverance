---
name: dead-reckoning
description: Use when running, inspecting, or comparing Dead Reckoning model traces, especially after tensor, sampler, quantization, native-kernel, or inference changes.
---

# Dead Reckoning Skill

Use Dead Reckoning to detect model-runtime regressions without treating generated prose as an exact
oracle.

## Problem To Solve

Runtime changes are not uniformly risky across models. One change may produce negligible drift for
Qwen while causing large drift for Gemma, TinyLlama, Nemotron, or Antares. The agent must test the
models separately and report model-specific behavior.

Generated answers are also weak regression signals. The same question can have several valid answers,
and a six-token smoke run may end before the answer is complete. Use generated text to detect obvious
corruption or gross behavioral changes, not to demand one exact answer unless the test explicitly has
an invariant output.

Exact cross-environment inference reproducibility is not currently solved. Mac and Colima/Linux ARM
may use different native libraries, JIT paths, SIMD behavior, or compiler settings. Two runs on the
same computer with the same artifact, prompt, seed, and runtime have been observed to be identical;
use that as the strongest current reproducibility claim.

## Run The Smoke Test

The current test is:

```text
core/src/test/java/io/teknek/reckoning/integration/DeadReckoningModelsIT.java
```

Non-generation task coverage is in:

```text
core/src/test/java/io/teknek/reckoning/integration/DeadReckoningTaskModelsIT.java
```

It exercises the commonly used embedding and classification models. Embedding builders inherit
`withDeadReckoning(...)`; sequence-classification builders expose the same method directly.

Run it through Failsafe. Do not use a normal `-Dtest` invocation for this `*IT` class:

```bash
MAVEN_OPTS="-XX:TieredStopAtLevel=1" mvn -pl core -Dit.test=DeadReckoningModelsIT \
  -Dfailsafe.failIfNoSpecifiedTests=false failsafe:integration-test failsafe:verify
```

The test must not use JUnit assumptions to hide a missing model cache. If a required model is absent,
the run should fail loudly. Record the generated trace paths from the test output.

## Compare Traces

Prefer the built-in Java check for same-environment equality:

```java
DeadReckoningTraceCheck.same(expectedTrace, actualTrace);
DeadReckoningTraceCheck.sameAsPrevious(currentTrace);
```

It returns `true` only when the semantic records are exactly equal. It returns `false` and prints a
summary containing missing records, unexpected records, or changed fields. JSONL line ordering is
ignored so parallel instrumentation does not create a false failure.

For each trace, inspect the JSONL records with `jq`. Start with the response and token sequence:

```bash
jq -r 'select(.kind=="RESPONSE_END") |
  {text, generated_tokens, finish_reason}' trace.jsonl
```

For custom generation paths that do not emit `RESPONSE_END`, use the test's printed
`DEAD_RECKONING_RESPONSE` line. Do not infer that a missing response event means generation failed.

Then compare checkpoint records by semantic identity, not file line number. The identity is the
checkpoint's phase, position, and name. Record ordering can change because model work is parallel or
because instrumentation observes execution order differently.

Useful checkpoint fields are:

- `dtype`, `shape`, and `count`: structural compatibility.
- `min`, `max`, `mean`, `sum`: distribution and bias changes.
- `mean_abs`, `max_abs`, `l2_norm`, `rms`, `stddev`: magnitude and energy changes.
- `zero_count`, `nan_count`, and infinity counts: numerical health.
- `hash`: exact captured-value identity, not a tolerance result.
- `first_values`: a quick local view of the actual numeric drift.

Task records must also be checked:

- `TASK_START`: task, input, and pooling type.
- `EMBEDDING_RESULT`: vector shape, dtype, statistics, hash, and first values.
- `CLASSIFICATION_RESULT`: every label score, selected label, and selected score.

Classification commonly emits both an embedding result and a classification result because the
classification path may first compute an embedding and then apply a classifier head. Compare both;
matching only the selected label is insufficient.

## Classify Differences

Use these categories in the report:

### Identical

The semantic checkpoint key and all relevant statistics match. Matching hashes provide the strongest
confirmation for same-environment reruns.

### Numerical Drift

Statistics differ, but shape, dtype, finite-value counts, overall scale, selected tokens, and finish
behavior remain stable. State the first drifting checkpoint and quantify representative deltas.

Example conclusion:

```text
Qwen dense and JQ4 first diverge at layer.0.query_projection.
Mean and RMS differ at approximately 1e-8, hashes differ, and all six selected tokens remain equal.
This is numerical/platform drift, not an observed behavioral regression.
```

### Material Tensor Change

Shape or dtype changes, non-finite values appear, magnitude changes substantially, or checkpoint
topology differs. Treat this as a likely runtime regression even if the final short text happens to
look reasonable.

### Behavioral Divergence

Selected token IDs, finish reason, response structure, or model-specific control flow changes. This
is more important than a hash difference and should be reported prominently.

For task models, behavioral divergence includes changed embedding shape, changed label set, changed
selected class, or materially changed classification scores. Embedding vectors should be compared
exactly on the same machine and by statistics/hash across different environments.

### Incomplete Trace

The model returns successfully but the trace contains only checkpoints or lacks expected lifecycle
records. Report that limitation separately. Nemotron's custom AR and diffusion paths currently need
the printed response line in addition to the JSONL trace.

## Same-Machine Reproducibility Check

When comparing two runs from the same machine:

1. Confirm identical model artifact, prompt, seed, temperature, and token limit.
2. Count records and checkpoint records.
3. Match checkpoints by phase, position, and name.
4. Compare all statistics and hashes.
5. Compare selected tokens and final response.
6. Treat a mismatch as a real regression candidate until explained.

Do not call two traces different merely because their JSONL byte hashes differ. First normalize or
sort records by semantic checkpoint identity. Ordering differences alone are not numerical drift.

## Cross-Environment Comparison

When comparing Mac with Colima/Linux ARM:

- Expect floating-point statistics and hashes to differ.
- Find the first changed checkpoint.
- Report absolute and relative scale of the change.
- Check finite counts, shapes, dtypes, selected tokens, finish reason, and response behavior.
- Do not require exact hashes across environments.

For dense HF versus Reckoning traces, use the threshold checker rather than exact equality:

```java
DeadReckoningCrossRuntimeCheck.Result result = DeadReckoningCrossRuntimeCheck.compare(
        hfTrace,
        reckoningTrace,
        DeadReckoningCrossRuntimeCheck.Policy.causalDefaults());
```

Interpret the result as follows:

- `GREEN`: all mapped stages are within pass thresholds.
- `YELLOW`: numerical warnings exist, but no hard structural or threshold failure occurred.
- `RED`: a required mapped stage is missing, shapes are incompatible, token IDs differ, or values
  exceed warning thresholds.

The checker maps common HF Qwen/Llama/Gemma stage names to Reckoning names and treats a leading
single batch dimension as compatible with Reckoning's flattened sequence shape. Add model-specific
stage mappings and thresholds for architectures such as Gemma4 rather than weakening the global
thresholds.

The most useful report is concise and ordered:

1. Whether both runs completed.
2. First divergent checkpoint.
3. Representative statistic deltas.
4. Whether token selection changed.
5. Whether final behavior changed.
6. Any missing or incomplete trace events.

Never claim that a model answer is wrong solely because it differs from another plausible answer. The
agent should distinguish numerical drift, changed sampling, incomplete output budget, and actual
runtime corruption.
