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

## GPU Redesign Contract

This section is the normative design for the next GPU implementation. The current
implementation is a staging implementation and must not be treated as the target
architecture.

### Device Tensor Ownership

`TensorRef.device()` identifies the storage location of every operand. A GPU tensor
must have a nonzero `deviceHandle()` and does not need a host `MemorySegment`.
`memorySegment()` is a CPU-storage API and must not be used by GPU providers.

Every operation must validate that all tensor operands are on the same device before
provider selection. CPU/GPU mixing is not implicit. Transfers are explicit through
`Lighter.to(source, targetDevice)`.

### Quantized Sidecars

I8 and Q4 tensors consist of two device allocations:

- Packed payload.
- F32 quantization-scale sidecar.

Allocation, transfer, pooling, and release must preserve both allocations. A GPU
quantized tensor without its matching sidecar is invalid and must be rejected before
dispatch.

`Lighter.to(...)` must transfer the payload and sidecar as one ownership unit. A
successful transfer returns a complete TensorRef; partial transfers must clean up all
already-created allocations.

### Tensor Layout and Strides

Tensor shape and storage layout are separate concepts. For every 2D tensor:

- `stride` is the physical distance between row starts in logical elements.
- A one-row tensor still has a nonzero stride equal to its row width.
- Q4 payload byte offsets are logical element offsets divided by two.
- I8, BF16, F16, and F32 payload offsets use their dtype byte width.
- Scale offsets use the dtype-specific block index, not payload byte offsets.

GPU ABI calls must receive logical offsets, byte offsets, and strides in explicitly
named parameters. No ABI function may infer these values from a nullable host
memory segment.

### GPU GEMM ABI

The normal Tensor2 GPU GEMM contract is:

```text
GPU input + GPU scales + GPU Q4 weights + GPU weight scales -> GPU result
```

The ABI must accept:

- Input buffer handle.
- Input scale buffer handle when applicable.
- Weight buffer handle.
- Weight scale buffer handle.
- Result buffer handle.
- Logical row/column offsets.
- Physical strides.
- Matrix dimensions.
- Kernel kind.

The ABI must return a status. Invalid handles, ranges, unsupported kernels, command
encoding failures, and submission failures must be observable by Java. A `void`
entrypoint that silently returns on error is not acceptable for the V2 provider.

### Command Completion

Returning from a native GPU method does not imply that GPU work is complete. Every
operation must expose a completion boundary before its result is read or its buffers
are released.

The required sequence is:

1. Encode the command.
2. End every pass and finish the command encoder.
3. Submit the command buffer.
4. Await queue completion or an operation fence.
5. Only then allow readback or resource release.

The result of a GPU GEMM must be readable through an explicit GPU-to-CPU transfer
after this completion boundary.

### Provider Capabilities

Providers must advertise exact support. GPU support is not equivalent to generic
Tensor2 support.

Each provider capability must specify:

- Input dtype.
- Weight dtype.
- Result dtype.
- Supported kernel kinds.
- Alignment requirements.
- Supported device.
- Whether the operation is synchronous or asynchronous.

For example, if a native shader implements only batch GEMM, `GpuOps` must reject
`ROW` and `CHUNK` kernels. It must not claim support and produce an incomplete or
zero result.

### Host-Staging Operations

Some GPU operations may intentionally stage CPU data through the GPU. Those are a
separate operation category and must be explicitly named as host-staging operations.
They must not be exposed as the normal GPU Tensor2 provider contract.

The normal `GpuOps` provider is device-pure. A future staging provider must document:

- Which operands remain on CPU.
- Which transfers occur.
- Whether the operation blocks for completion.
- Which buffers are shared or pooled.
- The expected transfer cost.

### Resource Pools and Concurrency

Per-operation resources should eventually be leased from a pool:

- Input buffers.
- Scale buffers.
- Parameter buffers.
- Result buffers.
- Readback buffers.
- Command encoder/submission state.

The lease remains live until GPU completion. It may not be returned when the Java
downcall returns if the command is still in flight.

Once resources are per-operation and completion is explicit, multiple operations
may be in flight. Until then, the shared-runtime lock described above remains the
correct safety boundary.

## Required GPU Test Matrix

The GPU test harness must create CPU oracle tensors, explicitly transfer complete
operands to GPU, execute the GPU operation using GPU TensorRefs, transfer the result
back to CPU, and compare against the CPU oracle.

The harness must be reusable and cover:

- F32 x Q4 -> F32 where supported.
- I8 x Q4 -> F32.
- One-row and multi-row inputs.
- One-row and multi-row result ranges.
- Input offsets and tails.
- Q4 weight offsets and tails.
- Result offsets.
- Q8 and Q4 sidecars.
- BATCH, ROW, and CHUNK only when the provider advertises them.

Unsupported provider/dtype/kernel combinations are assumptions or explicit skipped
cases. They are not silent successes. Supported cases must assert provider usage,
finite output, and numerical parity with the CPU oracle.

The minimum diagnostic test is a single `I8 x Q4 -> F32` case with `k=32`, followed
by the full fuzz matrix. The single case must validate payload transfer, sidecar
transfer, dispatch, completion, result copy, and readback before broad fuzzing.
