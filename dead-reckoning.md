# Dead Reckoning

Dead Reckoning is a short, structured inference trace used to detect behavioral and numerical
regressions in model-runtime changes.

It is designed for the practical problem we have while refactoring tensor execution:

- A code change may leave one model effectively unchanged while causing another model to drift
  substantially.
- Generated answers are not reliable regression oracles. Many prompts have multiple valid answers,
  and short generation may end before an answer is complete.
- Exact floating-point equality is not expected between different runtime environments, even when
  they use the same CPU architecture.

## Current Feature

Dead Reckoning is enabled on `AutoModelForCausaLm.Builder` with:

```java
.withDeadReckoning(new DeadReckoningParams(output, 32, selectors))
```

The runtime writes newline-delimited JSON records. Tensor checkpoints can capture:

- Model phase, position, layer, stage, and stable name.
- Tensor dtype, shape, and element count.
- Minimum, maximum, mean, sum, mean absolute value, and maximum absolute value.
- L2 norm, RMS, and standard deviation.
- Zero, NaN, positive-infinity, and negative-infinity counts.
- SHA-256 hash of the captured tensor values.
- A bounded sample of the first values.

The current model smoke test is `DeadReckoningModelsIT`. It runs eight six-token traces:

- Qwen3 0.6B dense and JQ4.
- TinyLlama JQ4 through the Deliverance model artifact.
- TinyLlama JQ4 through the second configured smoke case.
- Gemma2 JQ4.
- Nemotron 3B AR and diffusion modes.
- Antares 1B JQ4.

Traces are written outside Maven `target` so they can be reviewed and checked in:

```text
dead-reckoning/<hostname>/<model>-<timestamp>.jsonl
```

## What A Trace Proves

A passing trace run proves that the selected model loaded, executed its configured inference path,
and produced a response. It does not prove that the response is factually correct or that every
possible prompt will behave correctly.

For a fixed prompt, seed, model artifact, and runtime environment, two runs on the same computer
are currently repeatable. We have observed identical checkpoint statistics and hashes across such
runs, although JSONL record ordering can differ.

Repeatability across different environments is not solved. Mac and Colima/Linux ARM runs can have
different floating-point values and hashes while selecting the same tokens and producing equivalent
text. This is expected to be reported as numerical drift, not automatically as a functional bug.

Some model-specific generation paths, notably Nemotron's custom AR and diffusion paths, currently
emit checkpoint records without the complete standard response-event sequence. The test therefore
prints the returned response, token IDs, and finish reason separately.

## Running The Test

Run the focused integration test with Failsafe:

```bash
MAVEN_OPTS="-XX:TieredStopAtLevel=1" mvn -pl core -Dit.test=DeadReckoningModelsIT \
  -Dfailsafe.failIfNoSpecifiedTests=false failsafe:integration-test failsafe:verify
```

Do not add cache-based assumptions to this test. A missing required model should fail the test,
not appear as a passing or skipped model run.

## Interpreting Results

Use the trace to answer these questions in order:

1. Did the model load and complete generation?
2. Did the response and selected token sequence remain plausible and stable?
3. Are the checkpoint names, phases, positions, shapes, and dtypes aligned?
4. Where is the first changed checkpoint?
5. Is the change small numerical drift or a material change in the computation?
6. Did the drift change selected tokens, finish reason, or output behavior?

The first changed checkpoint is usually more useful than the final text. A small difference in an
early projection may propagate through later layers without changing the selected tokens. A change
in shape, dtype, checkpoint topology, non-finite counts, or token selection is more significant than
a changed hash alone.

## Exact Trace Check

Java callers can perform an exact semantic trace check. The check returns `true` when all records and
fields match, and returns `false` while printing a concise difference report otherwise:

```java
DeadReckoningTraceCheck.same(expectedTrace, actualTrace);
DeadReckoningTraceCheck.sameAsPrevious(currentTrace);
```

The check ignores JSONL record ordering, but it does not apply numeric tolerances. It requires exact
statistics, hashes, shapes, dtypes, tokens, responses, and lifecycle records. Use it for same-machine
reruns and same-environment migration gates. It reports the semantic checkpoint key and changed
fields when the result is false.
