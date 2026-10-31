---
name: faithful-port
description: Use when porting an implementation from another language or upstream repository and behavioral fidelity is required; compare source and target class-by-class, preserve semantics, and prove changes with fast tests.
---

# Faithful Port

Use this skill for ports from a source implementation such as Hugging Face Transformers into Deliverance. The source implementation is the behavioral authority. Do not call a structural approximation a faithful port.

## Core Rule

For every source class and method, create an explicit target counterpart. Compare the source on the left and the target on the right.

```text
SOURCE IMPLEMENTATION        TARGET IMPLEMENTATION
class/method                  class/method
variable                      variable
operation                     operation
shape                         shape
dtype                         dtype
cache ownership               cache ownership
```

## Naming

- Retain source class and method concepts.
- Convert naming idiomatically only where required: `hidden_states` becomes `hiddenStates`, `past_key_values` becomes `pastKeyValues`.
- Keep equivalent variables in the same scope and order.
- Do not hide a source operation inside an unrelated generic helper when a named target method makes the correspondence clearer.
- Name compound operations after their source behavior: `applyPle`, `applyMoeBranch`, `applyLayerScalar`, `updatePastKeyValues`.

## Nothing Is Omitted Silently

- Port every source branch, even when it appears unused by the first target checkpoint.
- Do not omit RoPE scaling, mask variants, cache branches, optional norms, dtype conversions, tied weights, or configuration fields because the current model does not seem to use them.
- If a branch is genuinely unsupported, retain a clearly documented target method or explicit failure and list it in the parity document.
- Do not replace behavior with a plausible shortcut.

## Performance And Correctness

- Correctness comes before performance.
- Do not replace an operation with a scalar Java loop merely because it is easier to implement.
- Use provider-backed tensor operations, Vector API/Panama implementations, chunking, or native acceleration where the target architecture supports them.
- A slow implementation is not faithful if it changes execution semantics or makes production use impractical.
- If the source uses an operation unavailable in the target, document the gap next to the replacement and add a characterization test at representative dimensions.

## Tests Before And After

Every ported behavior needs evidence at the smallest useful level.

1. Add a source test port with matching assertions when the source test maps to the target API.
2. Add a focused unit test for the mechanics: shapes, values, masks, cache ownership, dtype boundaries, and operation order.
3. Add a characterization/performance test for provider selection, vectorization, chunking, or allocation behavior when performance matters.
4. Capture the baseline before changing behavior when fixing a bug.
5. Rerun the same test after the fix.
6. Report the exact before/after result: failure becomes pass, tensor drift changes, allocation changes, throughput changes, or no measured change.

Do not write a test only after the fix and present its green result as proof that the fix helped. A green post-change test proves only that the new assertion passes.

## Reference Dumps

For model ports, generate a source reference dump and compare the target at named checkpoints:

- input IDs and attention masks
- embedding output
- every intermediate projection
- normalization output
- positional encoding output
- attention scores/probabilities/output
- residual outputs
- PLE or auxiliary streams
- MoE router scores, top-k indices, weights, and expert output
- persistent and temporary cache contents
- final normalized hidden state
- pre-sampling logits

Stop at the first numerical divergence. Do not diagnose from generated text alone.

## Cache Ownership

- Distinguish persistent request state from current-forward temporary state.
- Persistent cache state belongs to the request/session cache.
- Temporary shared state belongs to that same request/forward context.
- Immutable cache blocks may be shared with leases/refcounts.
- Never store request-specific mutable cache state on a model singleton.
- Test interleaved requests, cache reuse, cold replay, and session close/reopen.

## Documentation

Create a side-by-side Markdown document for each substantial port:

- source path and line range on the left
- target path and line range on the right
- aligned names and method structure
- exact tensor shapes and dtype transitions
- explicit unsupported/unproven items
- before/after test evidence
- remaining parity risks

Use words such as `implemented structurally`, `compiled`, `synthetically verified`, `source-parity verified`, and `production verified` precisely. Reserve `complete`, `faithful`, and `parity` for evidence-backed states.

## Completion Gate

Do not declare the port complete until:

- all source branches are represented or explicitly rejected;
- class and method comparisons are documented;
- focused unit tests pass;
- mapped upstream tests pass;
- provider/performance characterization exists where relevant;
- reference tensors/logits match for representative real checkpoints;
- cached and cold generation agree;
- final generated output is checked only after tensor parity is established.
