# vLLM-Style Decode Plan

## Goal

Build a decode path that keeps decode work on the GPU with one fused attention operation per layer/token, instead of Java orchestrating per-head/per-page GPU calls or reading intermediate attention scores back to CPU.

Target shape:

```text
decode token
  for each layer
    GPU-resident hidden state
    GPU QKV projection
    GPU RoPE + KV write
    GPU fused paged attention over all heads
    GPU output projection
    GPU MLP
  GPU output projection / sampler boundary
  CPU receives only final token or minimal logits metadata
```

This is not the current state. Current model execution is mixed CPU/GPU, and the current experimental GPU decode-attention implementation is not the desired performance path.

## Current Reusable Work

- `TensorOperations.decodePagedAttention(...)` exists as the provider boundary.
- `CausalSelfAttention` calls a provider primitive instead of owning all page loops.
- Native SIMD/Panama implementation works and is benchmark-stable.
- Native GPU tensor registry lifecycle now has:
  - `unregister_tensor(id)`
  - free-list slot reuse
  - registry bounds check
  - lookup/add-ref/release safety for registered buffers
- Tests exist for:
  - paged attention correctness
  - GPU registry unregister/reuse/exhaustion
  - Qwen GPU decode-attention path selection

## Current Throwaway Work

Do not optimize these paths further:

- GPU QK-only decode attention with CPU softmax/value accumulation.
- Per-head packed K/V GPU attention.
- Per-head GPU launch/readback loop.
- CPU packing K/V per head.

These preserve the wrong bottleneck:

```text
36 layers * 32 heads = 1152 GPU launches/readbacks per token
```

They are useful only as evidence/tests, not as the product path.

## vLLM Reference Shape

vLLM decode attention consumes:

```text
Q
K_Buffer
V_Buffer
Req_to_tokens page table
B_Seqlen
```

The kernel directly indexes paged KV:

```text
page = req_to_tokens[token / page_size]
row  = token % page_size
K[page, row, kv_head, dim]
V[page, row, kv_head, dim]
```

It performs inside GPU:

```text
QK score
scale / softcap
online softmax
V accumulation
write attention output
```

It does not:

- register KV tensors in the hot loop
- repack the full KV window per token
- read attention scores back to CPU
- launch per head from Java

## Required Architecture

### 1. GPU Resource Lifecycle

Keep the low-level registry lifecycle. It is required but not sufficient.

Native API:

```c
int64_t register_tensor(const char *data, int size);
void unregister_tensor(int64_t id);
WGPUBuffer lookup_tensor_addref(int64_t id);
void release_tensor(WGPUBuffer buffer);
```

Rules:

- Model weights are persistent registrations.
- Runtime tensors must not leak registrations.
- GPU ops borrow buffers with add-ref/release.
- Registry overflow must fail loudly, never corrupt memory.

### 2. GPU KV Page Pool

Add a GPU-backed KV page pool instead of registering Java KV page tensors during decode.

Required concepts:

```text
GpuKvCache
  keyBuffer: GPU buffer for all key pages
  valueBuffer: GPU buffer for all value pages
  pageSize
  kvLength
  pageCount
```

Each KV page has stable physical page id:

```text
logical cache page -> physical GPU page index
```

KV page lifecycle:

- allocate page
- write new K/V row
- reuse/free page
- release page at session/cache shutdown

Do not allocate/register/free GPU pages per attention call.

### 3. GPU Page Table

For a decode request/layer, provide the GPU kernel with a compact page table:

```text
int pageTable[numPages]
```

This is equivalent to vLLM `Req_to_tokens` for the single sequence case.

Initial implementation can upload this table each decode attention call. Later it can be cached/reused.

### 4. All-Heads Fused Decode Attention Kernel

Add native API:

```c
void gpu_decode_paged_attention(
    int64_t scratch_id,
    int64_t shader,
    const float *query,
    int query_offset,
    int64_t key_buffer_id,
    int64_t value_buffer_id,
    const int *page_table,
    int page_count,
    int visible_rows,
    int page_size,
    int number_of_heads,
    int number_of_kv_heads,
    int head_size,
    int kv_length,
    float scale,
    float *output,
    int output_offset
);
```

WGSL grid:

```text
workgroup_id.x = head
```

Kernel logic:

```text
head = workgroup_id.x
kvHead = head / headGroupSize

for token in 0..visibleRows:
  page = pageTable[token / pageSize]
  row = token % pageSize
  score = dot(query[head], keyBuffer[page, row, kvHead])
  online softmax update
  acc += probability * valueBuffer[page, row, kvHead]

output[head] = acc / sum
```

Properties:

- one native call per layer/token
- one GPU dispatch for all heads
- one output readback for `valueOut` until decode tensors become GPU-resident
- no per-head Java loop
- no per-page Java loop
- no attention score readback

### 5. Provider Gate

`NativeGPUTensorOperations.supportsDecodePagedAttention(...)` returns true only when:

- GPU KV page pool exists for these pages
- query/output dtypes are supported
- softcap is supported, or softcap is null
- page table is available
- scratch/workspace sizes are safe

Otherwise return false and use primary provider.

Do not add user flags for normal behavior.

### 6. TensorPlan Integration

Do not block this work on full TensorPlan residency tracking.

Short term:

- Keep direct model call to `TensorOperations.decodePagedAttention(...)`.
- Implement provider-level resources below it.

Medium term:

- TensorPlan decode graph should use the same provider resources.
- TensorPlan should eventually track placement/residency:
  ```text
  CPU valid
  GPU valid
  authoritative placement
  provider handle
  version
  ```

## Implementation Phases

### Phase 1: Stabilize Current Code

- Keep `NativeGPUTensorOperations.supportsDecodePagedAttention(...)` conservative if the real GPU page pool is absent.
- Keep registry lifecycle fixes.
- Remove or quarantine per-head/per-page GPU paths from production selection.
- Keep tests that demonstrate why those paths are not the target.

Validation:

```sh
mvn -q -pl native -Dtest=NativeGPUSmallGemmIT test
mvn -q -pl native -Dtest=NativeGPUDecodePagedAttentionIT test
mvn -q -pl core -Dtest=Qwen3SmallIT#qwen34BJq4LoadsAndGeneratesShortAnswerWithGpuDecodeAttention test
```

### Phase 2: GPU KV Page Pool

- Add GPU KV buffer allocation for a fixed test shape first.
- Add Java/native API for uploading K/V rows into physical page slots.
- Do not integrate with model yet.

Tests:

- allocate page pool
- write rows
- read/compare rows via small diagnostic kernel or copyback
- free page pool

### Phase 3: All-Heads Fused Kernel Synthetic Test

- Implement `gpu_decode_paged_attention(...)` over GPU KV pool + page table.
- Test against Panama for Qwen-like shapes:
  - heads `32`
  - kvHeads `8`
  - headSize `128`
  - pageSize `32`
  - visibleRows `128`, `134`, `160`, `512`
  - full `4096` output comparison

No model involved.

### Phase 4: Model Integration

- Wire `KvBufferCache` page allocation to create/use GPU page pool when GPU provider is available.
- On KV write, update GPU page row.
- `CausalSelfAttention.decodePagedAttention` uses GPU only if page pool handles exist.

Tests:

- Qwen3 0.6B short GPU decode attention.
- Qwen3 4B JQ4 short GPU decode attention.
- Qwen3 4B JQ4 long generation GPU decode attention.

### Phase 5: Benchmark

Run normal benchmark, no special provider flag:

```sh
sh benchmarks/run-qwen-single-benchmark.sh
```

Expected counters:

```text
causalselfattention.decode_paged_attention.provider_Native_GPU_Operations
```

Compare:

- `tok_s`
- `causalselfattention.decode_paged_attention`
- `causalselfattention.score_value`
- allocator stats

## Non-Goals For Now

- Full TensorPlan residency tracking.
- Full GPU MLP decode.
- GPU sampling/top-k.
- Multi-sequence vLLM batching.
- Softcap in first kernel version unless Qwen config requires it.

## Hard Rules

- Do not force `--tensor-provider native-gpu` for decode attention experiments.
- Do not register transient KV tensors in the hot decode loop.
- Do not read back attention scores.
- Do not Java-loop GPU calls per head/page in production path.
- Do not add new user flags when local provider gating already fits the repo style.
- Every GPU resource allocation must have a corresponding lifecycle path.
