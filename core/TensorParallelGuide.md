# Tensor Parallel Guide

Tensor parallelism lets one generation request run across multiple rank-local model shards. The first supported and
tested model family is Gemma2.

Current support status:

* Supported: Gemma2, proven by `Gemma2TensorParallelIT`.
* Not yet supported: Llama, Mistral, Mixtral, Qwen2, GPT2, Gemma4.
* Recommended starting model: `tjake/gemma-2-2b-it-JQ4`.

## Requirements

Use a tensor-parallel size that cleanly divides the Gemma2 model dimensions. For `tjake/gemma-2-2b-it-JQ4`, use `1`, `2`,
or `4` ranks.

The current integration test uses:

* 4 tensor-parallel ranks.
* 2 physical gossip nodes.
* Up to 2 ranks per physical node.
* HTTP rank transport.
* HTTP collectives.

## Deployment Spec

Create a `TensorParallelDeploymentSpec` with:

* `deploymentId`: logical deployment name used for gossip keys.
* `requestedNodes`: tensor-parallel rank count.
* `maxRanksPerNode`: maximum ranks a physical node may host.

Example:

```java
TensorParallelDeploymentSpec deploymentSpec = new TensorParallelDeploymentSpec("gemma2-demo", 4, 2);
```

## Node Setup

Each physical node builds an `AbstractModel` with matching `GossipParallelSettings`.

```java
ModelFetcher fetcher = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");

AbstractModel nodeModel = AutoModelForCausaLm.newBuilder(fetcher)
        .withParallelSettings(new GossipParallelSettings(
                clusterName,
                nodeId,
                nodeUri,
                seedMembers,
                gossipSettings,
                deploymentSpec))
        .build();
```

Every node in the deployment must use the same `clusterName`, seed list, and `deploymentSpec`.

When `withParallelSettings(...)` is present, `build()` starts the node's gossip membership and attaches it to the model:

```java
GossipParallelMembership membership = nodeModel.gossipParallelMembership().orElseThrow();
```

## Runtime Lifecycle

Membership, leader election, assignment, collectives, and rank workers are now implementation details started from the
model build. The user does not need to call `startParallelMembership()`, `voteForLeader()`, `publishAssignmentAsLeader()`,
or `TensorParallelWorker.start(...)` directly.

The automatic lifecycle is:

```text
        node model build()                         gossip shared state
  +----------------------------+              +-------------------------+
  | AutoModelForCausaLm.Builder|              | deployment spec         |
  |   withParallelSettings     |              | candidates              |
  +-------------+--------------+              | leader vote             |
                |                             | assignment              |
                v                             | collective URI          |
  +----------------------------+              | rank endpoints          |
  | AbstractModel              |              +-----------+-------------+
  |  owns GossipMembership     |                          ^
  +-------------+--------------+                          |
                |                                         |
                v                                         |
  +----------------------------+                          |
  | GossipParallelMembership   |--------------------------+
  |  waits for candidates      |
  |  elects leader             |
  |  publishes assignment      |
  |  leader starts collective  |
  |  starts local worker       |
  +-------------+--------------+
                |
                v
  +----------------------------+       +-------------------------------+
  | TensorParallelWorker       |       | HttpTensorParallelCollective  |
  |  starts rank HTTP servers  |<----->| Server (leader-owned)         |
  |  publishes rank endpoints  |       +-------------------------------+
  +-------------+--------------+
                |
                v
       rank-local transformer forward
```

## Run Generate

The coordinator still uses a normal non-rank model for tokenizer, prompt rendering, output projection, sampling, stop
handling, and response construction. The distributed transformer forward path is opened from the membership:

```java
GossipParallelMembership membership = nodeModel.gossipParallelMembership().orElseThrow();

try (TensorParallelGenerationGroup group = membership.openGenerationGroup();
     AbstractModel coordinatorModel = AutoModelForCausaLm.newBuilder(fetcher).build()) {
    var prompt = coordinatorModel.promptSupport().get().builder()
            .addUserMessage("What is 1 + 1?")
            .build();

    Response response = group.generate(
            coordinatorModel,
            prompt,
            new GeneratorParameters()
                    .withNtokens(64)
                    .withMaxTokens(16)
                    .withTemperature(0.0f),
            new DoNothingGenerateEvent());

    System.out.println(response.responseText);
}
```

The coordinator model is not a tensor-parallel rank. It provides tokenizer, output projection, sampler, stop handling,
and response construction. Transformer prefill and decode execution goes through the rank endpoints published by the
workers.

`membership.openGenerationGroup()` hides endpoint discovery and HTTP rank-client construction. The returned group owns the
rank clients and should be closed by the caller.

For instruct checkpoints, prefer `model.promptSupport().get().builder()` over `PromptContext.of(...)`. Raw prompts are
completion prompts and may cause the model to echo part of the user text before answering.

## Validation

Run the Gemma2 end-to-end integration test:

```sh
MAVEN_OPTS="-XX:TieredStopAtLevel=1" mvn -pl core -am \
  -Dtest=Gemma2TensorParallelIT \
  -Dsurefire.failIfNoSpecifiedTests=false \
  test
```

The test proves the full path: two gossip nodes, automatic leader election and assignment, leader-owned HTTP collectives,
automatic local worker startup, four HTTP rank endpoints, and `TensorParallelGenerationGroup.generate(...)` producing a
`Response`.

## Operational Notes

* Keep all nodes on the same model files and Deliverance build.
* Do not mix model families in one deployment.
* Use one deployment id per logical model deployment.
* Wait for all rank endpoints before serving traffic; `openGenerationGroup()` requires a completed assignment and rank
  endpoint publication.
* Treat non-Gemma2 model families as unsupported until they have explicit tensor-parallel tests.
