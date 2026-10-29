# Tensor2 and Lighter Migration Plan

## Objective

- No active migrated runtime path references `AbstractTensor`.
- Convert `AbstractTensor` references in each migrated family to `TensorRef`.
- Remove `TensorOperations` and its implementations from active migrated paths, while
  leaving V1 definitions available for deferred and not-yet-migrated behavior.
- Use `Lighter` as the operation layer for migrated paths.
- Convert all original C-based SIMD and GPU operations needed by migrated paths to new
  `Lighter` implementations; retain V1 implementations for deferred/legacy paths.
- Use `KVCache2` everywhere after all model families are converted.
- Audit KV cache usage method by method; legacy and `KVCache2` behavior may be mixed within
  the same classes.
- Remove every active legacy KV cache reference from each migrated method after it has been
  migrated to `KVCache2`; retain V1 cache code for deferred/unmigrated behavior.
- Remove V1 allocator/provider objects such as `ArrayQueueTensorAllocator` and
  `ConfigurableTensorProvider` from migrated execution paths and signatures; retain their
  definitions for V1 paths that have not yet been migrated.
- Make `Lighter` responsible for tensor2 operation resources and allocation in the migrated
  path.
- Audit operation/helper classes such as `LayerNorm` and `LayerNorm2` and identify every
  remaining V1 class that needs a tensor2/Lighter replacement.
- Add the required replacements before migrating model families; do not assume existing
  `*2` classes cover the complete operation surface.
- Treat Gemma3 and Gemma4 as incomplete model families during this migration; extended
  behavioral/performance testing for them cannot be performed until their ports are working
  and complete.
- Defer `gpuDecode`, the scratch-backed feature spanning GPU and CPU execution; it is not a
  migration completion gate for this effort.

## Scope

- Migrated runtime paths must contain no active V1 references.
- Tests, tools, examples, and benchmarks may retain references temporarily when their
  behavior has not yet been ported; convert them as equivalent behavior becomes available.
- `gpuDecode` remains preserved untouched and is outside this migration's active scope.
- Gemma3 and Gemma4 receive structural migration work now, with extended behavior/performance
  testing deferred until their ports are complete.
- For fully migrated families such as dense Qwen3, remove unused inherited
  `AbstractModel`/provider/allocator dependencies from active paths, constructors, and
  signatures rather than merely leaving them unused.
- Qwen3-MoE is in scope despite not currently being an active supported model. It must be
  ported to Tensor2/Lighter/KVCache2 and receive its corresponding tests.

## Model-Family Status

To be defined.

## Migration Workstreams

To be defined.

## Validation Strategy

At every migration stage, run these quick health checks in addition to the tests directly
related to that stage:

- `ClassificationTest`
- `DiskKvBackendTest`
- `Qwen3SmallIT`
- `Gemma2PromptIt#gemmaGuidedTest()`

Each stage should also run the relevant model-family, tensor2/Lighter, KVCache2, and
provider tests for the code being changed.

## Completion Criteria

- To be defined.

## Working Decisions

- Run all four quick health tests at each stage.
- If a quick test fails, investigate whether the failure is caused by the current stage or
  by a known unrelated/incomplete model. Ask for clarification when the cause is unclear.
- V1 code may remain in the repository; the completion requirement is that migrated active
  runtime paths no longer use it.
- `ModelSupport` and reflective model construction are legacy paths. Prefer explicit model
  builders/factories such as `AutoModelForCausalLM` and family-specific builders.
- Removing `ModelSupport` reflection is an eligible early workstream and should precede broad
  constructor-signature migration where practical.
- Add characterization, parity, and regression tests whenever they help the active migration.
  Treat characterization tests as temporary evidence unless they protect a documented
  contract; replace or remove stale/confusing tests instead of preserving them indefinitely.
- Every migration test should state whether it protects behavior, numerical parity, provider
  selection, cache lifecycle, or performance.
- `Lighter` owns tensor2 allocation and tensor-provider selection; migrated code should pass
  a `Lighter` rather than `ArrayQueueTensorAllocator` or `ConfigurableTensorProvider`.
- Tensor2/Lighter must not call or wrap the old native SIMD/GPU implementations. For each
  required operation, create the Lighter/Tensor2 implementation, prove it with a fuzz/parity
  test against the old operation, migrate callers, and then remove the old operation from
  the active migrated path. The old operation may remain for deferred or unmigrated paths.
