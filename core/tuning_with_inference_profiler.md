# Tuning With InferenceProfiler

This is a practical workflow for finding and validating inference optimizations in Deliverance.

The short version:

1. Run a stable benchmark with `--profile-stages`.
2. Pick a hot row that runs many times.
3. Make the smallest plausible optimization.
4. Add a focused correctness test.
5. Re-run the profiler and compare the same counters.
6. Keep the change only if the profile explains the win.

This document uses a real Qwen3-4B-JQ4 tuning session as the example.

## Run The Profiler

For Qwen3-4B-JQ4:

```sh
sh benchmarks/run-qwen4b-prefill-baseline-benchmark.sh
```

The script writes shareable artifacts under:

```text
benchmarks/runs/YYYY-MM-DD-githash[-dirty]-run-qwen4b-prefill-baseline-benchmark-HHMMSS/
```

It also prints per-turn profile rows such as:

```text
[profile] mlpblock.forward                              count=    9216 total_ms= 10704.311 mean_us= 1161.492
[profile] causalselfattention.forward                   count=    9216 total_ms=  9954.930 mean_us= 1080.179
[profile] mlpblock.gate_up_projection                   count=    9216 total_ms=  5770.510 mean_us=  626.140
[profile] mlpblock.down_projection                      count=    9216 total_ms=  3352.119 mean_us=  363.728
[profile] mlpblock.fused_activation_multiply_quantize   count=    9216 total_ms=  1327.858 mean_us=  144.082
```

The important columns are:

- `count`: how often the stage ran in this turn.
- `total_ms`: total wall-clock time attributed to the stage.
- `mean_us`: average time per call.

High `total_ms` tells you where request time goes. High `count` means small improvements can matter.

## Start From A Baseline

One baseline Qwen3-4B-JQ4 run showed throughput around 11 tokens/sec:

```text
old avg tok/s: ~11.45
```

The profiler made the broad shape clear:

```text
mlpblock.forward               hot
causalselfattention.forward    hot
sampler.output_projection      hot but only once per generated token
```

Inside MLP:

```text
mlpblock.gate_up_projection    very hot
mlpblock.down_projection       hot
mlpblock.down_quantize         repeated every layer/token
```

`mlpblock.down_quantize` looked attractive because it ran once per layer per forward:

```text
36 layers * 256 forwards = 9216 calls
```

At roughly `160us` per call, a small win could become hundreds of milliseconds per request.

## Try The Obvious Fusion First

The MLP shape is:

```text
gate = input @ gateWeight
up = input @ upWeight
hidden = activation(gate) * up
hiddenQ = quantize(hidden)
output = hiddenQ @ downWeight
```

The old path materialized `hidden` as F32 and then quantized it:

```text
pass 1: activation(gate) * up -> F32 hidden
pass 2: quantize F32 hidden -> Q8 hiddenQ
```

The better target was:

```text
activation(gate) * up -> directly write Q8 blocks
```

That removes a full intermediate pass and avoids treating the F32 post-activation tensor as a long-lived result.

## Java First

The first implementation added a `TensorOperations` primitive:

```java
activationMultiplyQuantize(gate, up, activation, DType.I8, offset, length)
```

The Panama implementation preserved the existing Q8 layout:

```text
for each row:
  for each 32-value block:
    compute activation(gate[i]) * up[i]
    find max abs
    write one Q8 scale
    write 32 quantized bytes
```

Correctness test:

```text
separate activation*multiply + existing quantize
vs
fused activationMultiplyQuantize
```

The Java fused path changed the profile from a separate quantize stage to:

```text
mlpblock.fused_activation_multiply_quantize mean_us=144.082
```

`mlpblock.down_quantize` disappeared from the gated MLP path.

That was already useful.

## Where TensorPlan Fits

The long-term goal is not to collect one-off helper methods forever. The goal is to make tensor code read like one logical plan, then let the runtime choose better physical implementations.

The old style tends to sprawl:

```java
dotProductBatchChunk(... gate/up ...)
activation loop
multiply loop
quantize
dotProductChunk(... down ...)
```

That makes it hard to see the real operation:

```text
down(quantize(activation(gate(input)) * up(input)))
```

`TensorPlan` is the TensorLib experiment to close that gap. It gives us a way to describe the logical tensor flow once and then attach different physical strategies underneath it.

Example TensorPlan output for a small MLP-shaped graph:

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

And for an explicit fused in-place path:

```text
└─ timer variablemlpblock.intstream_multiply -> gate [32x3072]
   └─ gate = fuse -> [32x3072]
      ├─ map 0 [ACTIVATION_MUL_IN_PLACE] gate = silu(gate) * up
      ├─ up [32x3072] F32 borrowed
      └─ gate [32x3072] F32 mutable
```

This is easier to review than scattered loops because it shows:

- tensor ownership: borrowed vs mutable
- logical names: `gate`, `up`, `hidden`
- operation shape: `[batch x hidden]`
- intended fusion point: `ACTIVATION_MUL_IN_PLACE`

The fused quantize work should move further in this direction. Ideally the model code eventually says something close to:

```java
Tensor hiddenQ = plan.input("gate", gate)
        .activate(SILU)
        .multiply(plan.input("up", up))
        .quantize(I8)
        .as("hiddenQ");
```

Then physical lowering can pick:

```text
Java fallback
Panama Vector implementation
Native SIMD implementation
GPU implementation
```

without rewriting model code each time.

That is why TensorLib matters: it gives contributors one clear path for tensor work instead of a collection of bespoke loops, benchmark scripts, and handwritten temporary tensors. The immediate optimization may still land as a `TensorOperations` primitive, but the direction is to make those primitives visible as TensorPlan operations.

## Then Native

The next version added a native C kernel:

```c
activation_multiply_quantize_silu_q8(...)
```

The native path writes the Q8 bytes and sidecar scales directly.

Correctness test became parameterized over providers:

```text
Panama
Native SIMD
```

The native profile then showed:

```text
mlpblock.fused_activation_multiply_quantize mean_us=55.853
```

Compared with Java fused:

```text
144.082us -> 55.853us
~2.6x faster for that stage
```

On a full Qwen3-4B-JQ4 benchmark:

```text
old avg tok/s:          ~11.45
Java fused avg tok/s:   ~11.83
native fused avg tok/s: ~12.47
```

This was enough signal to keep the optimization.

## Do Not Trust One Number

End-to-end benchmark runs vary. Laptop power state, thermals, OS scheduling, and prompt differences can move tokens/sec around.

Use end-to-end throughput as the final sanity check, not the first source of truth.

Better evidence looks like this:

```text
same stage name
same or similar count
lower mean_us
lower total_ms
no correctness regression
```

For this optimization, the decisive row was:

```text
mlpblock.fused_activation_multiply_quantize
```

not the raw request-level `tok_s` alone.

## Failed Or Low-Signal Attempts Are Normal

Some attempts are educational but not decisive.

Example: fusing attention score scaling and softmax looked reasonable because the original path did multiple passes:

```text
scale(attn)
optional softcap(attn)
softmax(attn)
```

That can be improved mechanically, and replacing division with reciprocal multiply plus SIMD normalization is sensible. But the end-to-end signal was much harder to isolate. Attention score time moved around with run variability.

That does not make the idea bad. It just means it was not the clearest next optimization.

The MLP quantize path was clearer because:

- it had a hot profiler row
- it ran thousands of times
- it removed a concrete tensor pass
- it had a focused correctness test
- the profile row moved sharply after native implementation

## Pairing Workflow

This is a good area for contributors, even if they are not deep inference-engine experts.

A useful pairing loop is:

1. A person points at a profiler row and asks why it is hot.
2. AI traces the code path and proposes a small candidate change.
3. The person reviews the idea and rejects vague or risky guesses.
4. AI writes a focused test before or alongside the implementation.
5. The person runs the real benchmark on their machine.
6. The profile decides whether the change survives.

That is not magic. It is just tight engineering feedback.

Good tasks for this style:

- add a provider parity test
- move a repeated tensor pass into a fused primitive
- add a native kernel for a single well-scoped operation
- compare profiler rows before and after
- document the result, including misses

You do not need to be fearless. You need to be willing to inspect one hot loop, write one test, and let the profiler tell the truth.

## What To Look For Next

Good future candidates have these traits:

- high `count`
- high `total_ms`
- repeated full passes over the same tensor
- a clear existing reference implementation
- a small output surface that is easy to test

Examples:

```text
MLP activation/multiply/quantize
attention score transform/softmax
RMSNorm row reduction/write
QKV projection packing
sampler output projection
```

Avoid starting with huge rewrites. The best wins in this codebase usually come from one narrow path that the profiler proves is hot.
