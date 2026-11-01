# GPU Concurrency

## Current Model

The current WebGPU implementation is serialized at the Java provider boundary for
the Tensor2 GPU path. `GpuOps.batchDotProduct()` holds the class lock for the
entire operation:

1. Register weight and scale buffers.
2. Copy inputs into shared native scratch buffers.
3. Create and encode the command buffer.
4. Submit the command buffer and wait for readback completion.
5. Unregister the temporary buffers.

This is intentionally a complete-operation lock. Locking only command submission
would be insufficient because the operation also uses process-wide scratch input,
parameter, result, and readback buffers.

The legacy `NativeGPUTensorOperations` path has a separate `GPU_SUBMIT_LOCK` around
its GEMM and attention submissions. That lock protects the legacy provider's
submission paths, but it is not the same lock as `GpuOps.class`.

## What Is Safe Today

- Tensor2 GPU batch GEMMs are serialized against other Tensor2 GPU batch GEMMs.
- Legacy GPU GEMM and decode operations are serialized against other operations
  using the legacy provider lock.
- GPU initialization is separately synchronized.
- GPU buffer allocation and tensor registration have their own lifecycle locks,
  but are not covered by one global GPU-operation lock.

The GPU hardware is not inherently limited to one operation. The limitation is the
current implementation's shared WebGPU/Dawn state and scratch storage.

## Known Boundary

The Tensor2 and legacy providers use different Java locks while sharing the same
native WebGPU device and Dawn/Metal runtime. They must not be treated as a single
global concurrency domain. If both paths submit concurrently, macOS Metal can
abort with errors such as:

```text
A command encoder is already encoding to this command buffer
```

The current safe deployment rule is therefore one complete GPU operation in flight
across the shared runtime. The provider-specific locks are an incremental guard,
not a complete cross-provider solution.

## Future Directions

### Shared Runtime Lock

Introduce one lock owned by the native GPU runtime and use it from both Java
providers for all operations that touch the shared device, queue, registry, or
scratch state. This is the smallest correctness improvement and preserves the
current resource model.

### Per-Operation Scratch

Replace process-wide scratch buffers with an operation context containing:

- Input and auxiliary-input buffers
- Parameter buffer
- Result and readback buffers
- Command encoder and command buffer
- Temporary registered tensor handles
- Completion state or fence

Contexts can then be pooled and leased independently. The lease must remain valid
until GPU completion, not merely until Java returns from the downcall.

### Submission Queue

Add a native submission worker or queue that owns command encoding and submission.
Java callers would enqueue immutable operation descriptors and receive a completion
handle. This avoids calling Dawn from arbitrary parallel worker threads and gives
the runtime one place to manage resource lifetimes and errors.

### Multiple In-Flight Operations

After per-operation resources and completion tracking exist, allow multiple
contexts in flight. The scheduler can then overlap command encoding, transfers,
and GPU execution while enforcing dependencies for buffers that are actually
shared.

### Provider Unification

Make the legacy and Tensor2 providers use the same native runtime API and
submission coordinator. This removes the current split between `GPU_SUBMIT_LOCK`
and `GpuOps.class` and makes concurrency policy explicit rather than provider-
specific.

## Verification Requirements

Any concurrency change should verify:

- No Metal/Dawn command-encoder assertions on macOS.
- No races in shared scratch or readback buffers.
- Correct results with concurrent CPU callers.
- Correct cleanup after success, provider fallback, exceptions, and JVM test
  teardown.
- Tensor2 and legacy provider parity under the same workload.
- GPU fuzz and Nemotron GPU oracle coverage.
