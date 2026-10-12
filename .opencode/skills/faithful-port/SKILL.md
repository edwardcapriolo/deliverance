---
name: faithful-port
description: Use when porting or migrating an existing implementation between interfaces, tensor types, providers, or execution frameworks while preserving behavior and performance.
---

# Faithful Port

Use this skill whenever the task is to port existing code rather than redesign
it. Treat the existing implementation as the behavioral, scheduling, provider,
and performance specification.

## Required Standards

- Mathematical correctness must be proven within explicitly stated numerical tolerances.
- Mathematical equivalence alone is insufficient. A port that computes the same dot product but makes it materially slower is not faithful and moves the project backward.
- Preserve operation granularity, scheduling, provider selection, kernel selection, fallback behavior, memory layout, offsets, strides, and tuning behavior unless a deliberate deviation is requested and documented.
- Do not replace an existing execution plan with a simpler loop merely because the simpler loop is easier to implement.
- Static analysis showing that a capability is not active on one current code path is not a reason to omit its implementation.
- If a provider operation supports multiple dtype or shape permutations, implement all applicable permutations. If a permutation is genuinely unsupported, add a clear `TODO` comment describing the missing functionality, its constraints, and the affected path.

## Tests

- Every port must include short-running characterization tests using random data with the same shapes as the production path.
- Characterization tests must compare the old and new paths for output parity and report timings for the same warmed-up workload.
- Tests must cover relevant dtype permutations, offsets, strides, tails, row chunks, page boundaries, and non-divisible ranges.
- Definable behavior must be asserted in unit tests. This includes candidate selection, fallback selection, provider dispatch, split counts, and other observable policy.
- A test that only proves numerical output is insufficient when the port changes execution strategy.
- Do not hide required characterization behind new opt-in tags or system properties unless the existing project convention already requires that behavior.

## Missing Functionality

- Missing functionality is mandatory to document with a nearby `TODO` comment.
- The comment must identify what is missing, why it matters, which code path is affected, and what parity or performance behavior remains unverified.
- Do not silently fall through to a naive or scalar implementation when the old path used an optimized provider.

## Documentation

- Every method added or changed as part of the port requires Javadoc, including private methods.
- Javadoc must describe contracts, supported dtypes and shapes, offsets and stride assumptions, provider behavior, and fallback behavior where applicable.
- Private visibility is not an excuse to omit documentation.
- Remove duplication when adding shared port infrastructure, but do not collapse distinct execution paths if doing so obscures provider or scheduling differences.

## Review

- Every post-change review must be assigned to a subagent.
- The reviewer must independently inspect the old and new paths, provider dispatch, scheduling, tests, and performance evidence.
- Do not lead the reviewer with a prompt such as "I did this; verify it is correct." State the behavior that must be preserved and ask the reviewer to find deviations, missing permutations, missing tests, and performance risks.
- Do not claim a port is complete until the independent review has reported no unresolved findings or the findings have been fixed and re-reviewed.

## Completion Criteria

A faithful port is complete only when:

1. The old and new execution paths have been mapped side by side.
2. Provider operations and kernel choices are equivalent or explicitly documented as changed.
3. Randomized, production-shaped characterization tests prove output parity within tolerance.
4. Timings show no unexplained material regression.
5. Missing functionality has nearby `TODO` documentation.
6. Unit tests assert all definable behavior.
7. A subagent has independently reviewed the changes.

Never call a numerically equivalent rewrite a faithful port when its execution
strategy, provider path, or performance behavior differs without approval.
