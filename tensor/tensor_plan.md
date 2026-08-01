# TensorPlan

`TensorPlan` is a small lazy tensor workflow API for building readable tensor subgraphs and experimenting with local fusion rules.

It is not a general tensor compiler. The first target is the MLP block shape used by transformer models:

```text
gate   = input @ gateWeight
up     = input @ upWeight
hidden = silu(gate) * up
output = hidden @ downWeight
```

## Why It Exists

The older code expresses this as a sequence of eager tensor operations and loops:

```text
matmul gate
matmul up
full pass: activation(gate)
full pass: gate *= up
matmul down
```

That is simple, but it makes fusion awkward. If two operations touch the same tensor region, we want the option to execute them chunk-by-chunk:

```text
chunk 0: activation, multiply
chunk 1: activation, multiply
chunk 2: activation, multiply
```

instead of:

```text
all chunks: activation
all chunks: multiply
```

This can improve cache locality and reduce scheduling overhead. It also gives us a place to describe future physical plans such as GPU or native fused kernels without adding one-off methods to `TensorOperations` for every possible fusion.

## Ownership

TensorPlan is explicit about external tensor mutability.

```java
Tensor input = plan.input("input", inputTensor);
ImmutableTensor weight = plan.immutable("gateWeight", gateWeightTensor);
Tensor scratch = plan.mutable("scratch", scratchTensor);
```

- `input(...)` is borrowed read-only. TensorPlan must not mutate it.
- `immutable(...)` is borrowed read-only and intended for weights/constants. It cannot be used as a normal mutable expression target.
- `mutable(...)` is borrowed mutable. TensorPlan may mutate the caller-provided tensor.
- outputs of operations such as `batchDot(...)` are workflow-owned temporary tensors and may be reused or mutated by the physical executor.

This distinction matters because model weights must never be modified, while intermediate MLP buffers are safe to reuse.

## Basic API

```java
TensorPlan plan = new TensorPlan(ops, pool);

Tensor input = plan.input("input", lnemb);
ImmutableTensor gateWeight = plan.immutable("gateWeight", fullyConnectedWeights);
ImmutableTensor upWeight = plan.immutable("upWeight", upProjectionWeights);
ImmutableTensor downWeight = plan.immutable("downWeight", projectionWeights);

Tensor gate = input.batchDot(gateWeight).as("gate");
Tensor up = input.batchDot(upWeight).as("up");
Tensor hidden = gate.activate(ActivationFunction.Type.SILU).multiply(up).as("hidden");
Tensor output = hidden.batchDot(downWeight).as("output");

try (AbstractTensor result = output.materialize()) {
    // use result
}
```

The common operations lower to existing `TensorOperations` where possible:

- `batchDot(...)` uses `TensorOperations.dotProductChunk(...)`
- `multiply(...)` uses `TensorOperations.maccumulate(...)`
- `add(...)` uses `TensorOperations.accumulate(...)`
- `scale(...)` uses `TensorOperations.scale(...)`
- `quantize(...)` uses `TensorOperations.quantize(...)`

`activate(...)` is the current optimization gap. `TensorOperations` does not have a provider-backed activation primitive, so activation still uses a TensorPlan Java loop unless it is part of an explicit fused map pipeline.

## Explicit Fused Chunk Pipeline

For fusion experiments, use `fuse(...)`. This keeps the public API generic instead of adding methods such as `activateMultiply(...)` for every special case.

```java
Tensor hidden = plan.fuse("hidden", gate.shape())
        .read("gate", gate)
        .read("up", up)
        .map("hidden = silu(gate)", TensorPlan.TensorOp.SILU_WRITE,
                (ctx, offset, length) -> siluWrite(ctx.tensor("gate"), ctx.tensor("hidden"), offset, length))
        .map("hidden *= up", TensorPlan.TensorOp.MUL_IN_PLACE,
                (ctx, offset, length) -> multiplyInPlace(ctx.tensor("hidden"), ctx.tensor("up"), offset, length))
        .tensor();
```

Execution order is chunk-local:

```text
chunk 0:
  hidden = silu(gate)
  hidden *= up

chunk 1:
  hidden = silu(gate)
  hidden *= up
```

The `TensorOp` identifiers are intentionally explicit. They let future physical planners recognize common fused operations while keeping the Java mapper as a correctness fallback.

## Plan Output

`plan()` returns an ASCII tree intended for debugging and review.

Example:

```text
└─ hidden = multiply -> [2x2]
   └─ fused silu(gate) * up -> [2x2]
      ├─ activate SILU(gate) -> [2x2]
      │  └─ gate = batchDot -> [2x2]
      │     └─ batchDot(input, gateWeight) -> [2x2]
      │        ├─ input [2x3] F32 borrowed
      │        └─ gateWeight [2x3] F32 borrowed
      └─ up = batchDot -> [2x2]
         └─ batchDot(input, upWeight) -> [2x2]
            ├─ input [2x3] F32 borrowed
            └─ upWeight [2x3] F32 borrowed
```

The output avoids internal `TensorShape` details such as sparse ranges unless we later decide those details are useful for physical planning.

## Metrics

Individual expression nodes can be timed:

```java
Tensor hidden = gate
        .activate(ActivationFunction.Type.SILU)
        .multiply(up)
        .timer("mlpblock.multiply")
        .as("hidden");
```

For explicit fused pipelines, timers can be attached to individual map steps:

```java
.map("hidden = silu(gate)", TensorPlan.TensorOp.SILU_WRITE, "mlpblock.silu", mapper)
.map("hidden *= up", TensorPlan.TensorOp.MUL_IN_PLACE, "mlpblock.multiply", mapper)
```

The current timer support uses Dropwizard metric names so existing call sites can preserve current profiler output. Longer term, this should move behind a metrics abstraction with real tags.

## Replay Benchmark

Use the replay script to test the MLP fusion without loading a full model or running the inference profiler:

```sh
sh benchmarks/run-tensor-plan-mlp-replay.sh
```

Default output goes under a benchmark run directory:

```text
benchmarks/runs/YYYY-MM-DD-githash-script-time/tensor-plan-mlp-replay.csv
benchmarks/runs/YYYY-MM-DD-githash-script-time/tensor-plan-mlp-replay.json
```

Example override for Qwen 0.6B-like shapes:

```sh
TENSOR_PLAN_REPLAY_ARGS="--m-values 128,256,403 --hidden 1024 --intermediate 3072 --pool-size 4" \
  sh benchmarks/run-tensor-plan-mlp-replay.sh
```

Recent replay output:

```text
TENSOR_PLAN_MLP_REPLAY m=128 h=1024 i=3072 baseline_ms=911.205 plan_ms=535.453 speedup=1.7017 max_abs=0.00012207 mean_abs=0.00000055
TENSOR_PLAN_MLP_REPLAY m=256 h=1024 i=3072 baseline_ms=882.066 plan_ms=851.907 speedup=1.0354 max_abs=0.00000000 mean_abs=0.00000000
TENSOR_PLAN_MLP_REPLAY m=403 h=1024 i=3072 baseline_ms=1302.105 plan_ms=1289.966 speedup=1.0094 max_abs=0.00000000 mean_abs=0.00000000
```

Interpretation:

- the fused path is correct within F32 tolerance
- small prefill chunks can benefit from chunk-local fusion
- larger chunks are roughly neutral in the current CPU fallback implementation

## Current Integration

`MLPBlock` uses TensorPlan for the `activation + multiply` part of SwiGLU-style MLPs when `upProjectionWeights` is present.

The current integrated path preserves the existing metric name:

```text
mlpblock.multiply
```

It also increments:

```text
mlpblock.tensorplan.fused_activation_multiply
```

## Limitations

- TensorPlan is not a full tensor compiler.
- It does not yet model devices such as `cpu` or `gpu:0`.
- It does not yet have provider-specific lowering for fused activation kernels.
- `activate(...)` remains a Java-loop fallback because `TensorOperations` has no activation primitive.
- GPU fusion requires physical kernels that understand the `TensorOp` sequence; the current fused maps are CPU fallback logic.

The intended next step is to add device hints and provider-specific physical lowering without changing the logical API.
