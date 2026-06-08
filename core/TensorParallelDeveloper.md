# Tensor Parallel Developer Notes

This document summarizes the internal changes that support tensor-parallel generation in Deliverance.

Tensor parallelism is currently implemented and integration-tested for Gemma2. Other model families still need explicit
weight policies, shape validation, and end-to-end tests before they should be advertised as supported.

## Runtime Shape

Tensor parallelism splits a single model deployment into ranks. Each rank owns a local dense shard of the transformer
weights and local KV cache state. The coordinator runs the normal generation loop and calls every rank for each prefill or
decode step.

Important boundaries:

* The coordinator owns tokenization, output projection, sampling, stop handling, and `Response` construction.
* Rank services own transformer forward execution and rank-local KV state.
* Collectives combine per-rank partial results, currently through all-reduce sum.
* Gossip handles deployment membership, leader election, rank assignment, and rank endpoint discovery.

## Core Types

Planning and context:

* `TensorParallelContext` describes rank and tensor-parallel size.
* `StaticTensorParallelContext` is the current construction-time context implementation.
* `TensorParallelPlanner` validates that model dimensions can be cleanly split.
* `ShardRange` and `TensorParallelShardPlan` represent local shard ranges.

Weight loading:

* `TensorShardAxis` and `TensorShardSpec` describe row or column weight shards.
* `WeightLoader.load(String, TensorShardSpec)` loads only the local dense shard.
* `TensorParallelWeightPolicyResolver` maps semantic weight names to shard policies.
* `DefaultTransformerWeightPolicyResolver` covers common decoder-only transformer projections.
* `TensorParallelWeightLoader` applies policies while model code loads block weights.

Execution:

* `TensorParallelCollectives` is the collective abstraction.
* `SingleRankTensorParallelCollectives` is the non-TP implementation.
* `CausalSelfAttention` and `MLPBlock` use local shard dimensions and all-reduce where needed.
* `TensorParallelGenerationGroup` coordinates all rank endpoints for prefill, decode, and `generate(...)`.
* Each `TensorParallelGenerationGroup.generate(...)` call uses a fresh rank session id and closes that session on every
  rank when generation completes.

Membership and transport:

* `TensorParallelDeploymentSpec` describes one deployment: deployment id, tensor-parallel rank count, and maximum ranks
  per physical node.
* `GossipParallelMembership` publishes deployment state, candidates, leader votes, committed assignment, and rank
  endpoints.
* `TensorParallelWorker` builds all local assigned ranks and starts one HTTP rank server per rank.
* `HttpTensorParallelRankServer` and `HttpTensorParallelRankClient` expose `/batchForward` and `/forward`.
* `HttpTensorParallelCollectiveServer` and `HttpTensorParallelCollectives` provide HTTP all-reduce support.
* HTTP collectives add a deterministic per-rank sequence number to logical collective keys so repeated decode steps do not
  reuse the same wire key.
* `BinaryTensorPayloadCodec` is the runtime tensor wire format. Control messages are JSON; tensor bodies are binary.

## Gemma2 Integration

Gemma2 is the first complete path.

Implemented pieces:

* `Gemma2Model.loadTransformerBlockWeights()` loads sharded q/k/v/o and MLP gate/down/up projections.
* Attention uses local attention head, KV head, attention length, and KV length calculations.
* MLP computes rank-local partials and reduces output across ranks.
* KV cache allocation uses local KV length.
* Full generation is exercised by `Gemma2TensorParallelIT` using two gossip nodes, two workers, HTTP rank endpoints, HTTP
  collectives, and `TensorParallelGenerationGroup.generate(...)`.

Known Gemma2 split requirements for `tjake/gemma-2-2b-it-JQ4`:

* `num_attention_heads = 8`
* `num_key_value_heads = 4`
* `hidden_size = 2304`
* `intermediate_size = 9216`
* `head_dim = 256`
* Clean tensor-parallel sizes: `1`, `2`, and `4`

## Shard Contract

Tensor-parallel weight shards are local dense tensors. Callers should use local coordinates inside the shard, not sparse
global coordinates.

This matters because many runtime operations assume dense local tensors. Sparse global-coordinate semantics make common
matrix operations harder to reason about and were intentionally avoided for the tensor-parallel path.

Q4 shard constraints:

* Row sharding requires the column count to align with `Q4ByteBufferTensor.BLOCK_SIZE`.
* Column sharding requires the shard start and end to align with `Q4ByteBufferTensor.BLOCK_SIZE`.
* Q4 shards remain Q4; loading through F32 materialization would defeat the memory benefit.

## Generation Flow

`AbstractModel.generate(...)` remains the canonical single-model generation implementation.

Tensor parallel generation reuses the same logic through:

```java
AbstractModel.generateWithForwarder(..., GenerationForwarder forwarder)
```

The forwarder lets `TensorParallelGenerationGroup` inject distributed rank execution while the coordinator model reuses
normal tokenizer, sampler, stop-word, tool-call, and response code.

This keeps generation behavior in one place and avoids copy/pasting the token loop into the tensor-parallel package.

## Current Limits

* Gemma2 is the only model family with a full passing end-to-end tensor-parallel generation test.
* Runtime tensor payloads currently support `DType.F32`.
* Worker readiness state and assignment-hash readiness checks are still follow-up hardening work.
* Broader model support needs family-specific review. GPT2 packed QKV and MoE models are not expected to work without
  custom policies.

## Tests

Useful focused tests:

```sh
MAVEN_OPTS="-XX:TieredStopAtLevel=1" mvn -pl core -am \
  -Dtest=Gemma2TensorParallelIT \
  -Dsurefire.failIfNoSpecifiedTests=false \
  test
```

Transport and coordinator tests:

```sh
MAVEN_OPTS="-XX:TieredStopAtLevel=1" mvn -pl core -am \
  -Dtest=BinaryTensorPayloadCodecTest,HttpTensorParallelRankTransportTest,HttpTensorParallelCollectivesTest,HttpTensorParallelGenerationGroupTest \
  -Dsurefire.failIfNoSpecifiedTests=false \
  test
```
