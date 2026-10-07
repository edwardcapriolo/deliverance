# TensorRef Allocator Cleanup Plan

## First Safety Requirement

- Add an allocator-level `allowDirtyTensors(boolean)` policy.
- Default it to `true` to preserve current performance behavior.
- Add the corresponding model-builder option so the policy is configured before model execution begins.
- When the policy is `false`, every dirty allocation request must be zero-initialized anyway.
- The policy must apply centrally in the allocator, not only at individual call sites, so legacy and TensorRef paths receive the same protection.
- Add metrics distinguishing requested-dirty allocations from forced-zeroed dirty allocations.
- Add a test proving a dirty request cannot return stale pooled contents when `allowDirtyTensors(false)` is active.

## Do Now

### 1. Define Allocation Semantics

- `Lighter.allocate(...)` returns zeroed logical storage, matching legacy `TensorAllocator.get()`.
- Add `Lighter.allocateDirty(...)` for write-only storage, matching legacy `getDirty()`.
- Dirty storage is undefined until the caller overwrites every element it will read.
- Audit current allocation call sites and use dirty allocation only where complete initialization is guaranteed.

### 2. Define Borrowed Storage

- Borrowing never zeroes or clears the underlying storage.
- Closing a borrowed ref never returns its payload to the allocator.
- `clear(...)` is always an explicit mutation of the referenced storage; it must not happen implicitly during borrow or close.
- Add tests proving borrowed views do not alter or return parent storage unexpectedly.

### 3. Make Zeroing Dtype-Correct

- Dense `F32`, `BF16`, and `F16` can use byte zeroing.
- `I8` must clear payload and scale metadata.
- `Q4` must use the logical zero encoding, not raw zero bytes. Packed zero values require nibble `8` (`0x88` for a full byte), with valid scale metadata.
- Add tests that verify logical values after clearing, not only raw bytes.

### 4. Add Core Metrics

Track through the allocator metric registry:

- allocation requests by dtype and mode
- pool hits and misses
- new versus reused payloads
- zeroing count, bytes, and elapsed time
- quantized sidecar allocation/reuse

Use stable metric names and low-cardinality dtype/mode tags. Do not create metric names per tensor shape.

## Defer

- Global retained-memory capacity and eviction policy.
- Device-specific allocator pools beyond CPU.
- Advanced ownership diagnostics beyond existing use-after-close and double-close checks.

## Completion Criteria

- Allocator tests cover zeroed allocation, dirty allocation, pooled reuse, borrowed views, and quantized zeroing.
- TensorRef model paths explicitly choose zeroed versus dirty allocation.
- Metrics show allocator behavior during focused TensorRef model tests.
