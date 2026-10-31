---
name: behavior-fix
description: Use when correcting existing behavior or a regression; establish the current behavior with a fast green test, change the assertion to the intended behavior so it goes red, then fix production code and return the test to green.
---

# Behavior Fix

Use this workflow for behavior changes, bug fixes, and regressions. Do not change production code before the baseline behavior is characterized.

## Required Sequence

### 1. Characterize Current Behavior

- Add the smallest fast unit, mock, or characterization test.
- Assert the behavior the code currently produces.
- Run the test and require it to pass green.
- Record the exact current output, shape, tokens, exception, or metric.
- Do not use a slow end-to-end test when a component test can observe the behavior.

### 2. Define the New Contract

- Change only the assertion or expected fixture to the intended behavior.
- Run the same test again.
- Require it to fail red for the expected reason.
- If it does not fail, the test does not prove the bug or the new contract is not different.

### 3. Fix Production Code

- Make the smallest production change that addresses the identified cause.
- Do not weaken the new assertion.
- Do not add compatibility behavior unless required by an existing consumer or persisted contract.
- Keep unrelated changes out of the fix.

### 4. Verify Green

- Run the same focused test and require it to pass.
- Run adjacent focused tests for the changed component.
- Report the three states explicitly:
  - baseline green result
  - changed-contract red result
  - fixed-code green result
- Report the exact behavior change, not only the files changed.

## Test Selection

Prefer, in order:

1. Pure unit test.
2. Provider/component test with a deterministic fixture.
3. Mock-based boundary test.
4. Small integration test only when the boundary is the behavior under test.
5. Full model or end-to-end generation test only after the component contract is green.

## Prohibited Shortcuts

- Do not patch code first and retrofit a green test.
- Do not report a post-fix green test without the baseline red transition.
- Do not infer correctness from non-empty output, compilation, or lack of exceptions.
- Do not replace a precise expected value with a weak assertion such as non-null, finite, or non-blank unless that is the actual contract.
- Do not call a structural change faithful without source behavior and test evidence.
