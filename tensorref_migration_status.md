# TensorRef Migration Status

## Completed

- Migrated the KV v2 storage path to `TensorRef`, including dense storage, disk persistence, sessions, read views, and TurboQuant boundaries.
- Fixed TensorRef multidimensional slice base offsets.
- Fixed TensorRef-backed KV page ownership and repeated page-view reuse.
- Added sidecar-aware TensorRef copy operations for I8 and Q4.
- Added `Lighter.clear(...)`, clearing payloads and quantization sidecars.
- Added builder-based Lighter operations:
  - `Accumulate`
  - `ArgMax`
  - `Exp`
  - `Max`
  - `Sum`
  - `Transpose`
  - materializing `Split`
- Added Naive and Panama provider implementations plus focused parity/fuzz tests for these operations.
- Added Tensor2 native `exp_f32` and `F32 x BF16` batch-dot ABI paths with regenerated jextract bindings.
- Added TensorRef entropy support in `TensorProbability`.
- Added TensorRef versions of diffusion-gemma entropy/stopping boundaries.
- Added TensorRef `TensorPlan` inputs, immutable inputs, planned values, add-node execution, easy arithmetic nodes, MLP execution, and fused Ref callbacks.
- Migrated model embedding implementations to return TensorRefs.
- Added direct TensorRef KV v2 forward and batch-forward entrypoints.
- Added `GenerationBackendRef`, `GenerationEngineRef`, TensorRef sampler/output projection, logits processing, XTC, scaling, and argmax paths.
- Added an end-to-end tiny Qwen3 TensorRef generation test.

## Verification

- Qwen3 TensorRef attention parity passes.
- Qwen3 ported-model parity passes.
- KV session, prefix-cache, disk persistence, TurboQuant packing, and slice/quantization tests pass.
- Tensor2 operation fuzz suites pass for the current implemented providers.
- Tiny TensorRef Qwen generation produces coherent text.

## Remaining Migration

- `AbstractModel` and generation APIs still expose legacy `AbstractTensor` paths alongside the new Ref path.
- `GenerationEngine` and the legacy GPU decode path remain AbstractTensor-based by design; the GPU path needs a separate TensorRef scratch/ownership design before porting.
- TensorPlan still has legacy-only nodes and needs Ref implementations for remaining wrappers/chunk/fused variants.
- DiffusionGemma model fields, loading, encoder, self-conditioning, attention, and generation body still use AbstractTensor extensively.
- Remaining model families, embeddings, classifiers, tensor-parallel transport, and legacy generator classes still use AbstractTensor.
- TensorRef output loaders are not implemented for every model family.
- `F32 x Q4` and `I8 x Q4` Tensor2 GEMM symbols exist in the ABI, but their bodies still delegate to row kernels rather than true tiled GEMM implementations.
- The Tensor2 `F32 x BF16` native GEMM path has an independent vectorized ARM/AVX implementation, but the full Q4 tiled kernel parity/performance work remains.

## Performance Status

- Historical Qwen3-0.6B throughput reached approximately 30-32 decode tokens/second on the host.
- The TensorRef benchmark path initially fell below that because output projection and Tensor2 dtype kernels were falling back or using row-oriented implementations.
- The TensorRef sampler output-projection bottleneck was reduced by adding native Tensor2 BF16 support and parallel output chunks.
- Current performance still requires a same-host rerun after the latest native kernel changes.
- Benchmark scripts must distinguish legacy `model.generate(...)` runs from the TensorRef `generateWithBackendRef(...)` path.

## Rules

- Do not add compatibility overloads to hide an incomplete migration.
- Do not link Tensor2 providers to the legacy `NativeSimd` implementation.
- `NaiveOps` may use scalar loops.
- `PanamaOps` must use Vector API implementations; do not add ordinary Java loops there without proving equivalent performance.
- Tensor2 native operations require their own C ABI, implementation, jextract binding, provider dispatch, and parity coverage.
- Stop and discuss any genuinely missing TensorRef/Lighter primitive before inventing a replacement.
