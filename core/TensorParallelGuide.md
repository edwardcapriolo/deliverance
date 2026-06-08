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

Each physical node creates an `AutoModelForCausaLm.Builder` with matching `GossipParallelSettings`.

```java
ModelFetcher fetcher = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");

AutoModelForCausaLm.Builder nodeBuilder = AutoModelForCausaLm.newBuilder(fetcher)
        .withParallelSettings(new GossipParallelSettings(
                clusterName,
                nodeId,
                nodeUri,
                seedMembers,
                gossipSettings,
                deploymentSpec));
```

Every node in the deployment must use the same `clusterName`, seed list, and `deploymentSpec`.

## Membership And Assignment

Start gossip membership on every node:

```java
GossipParallelMembership membership = nodeBuilder.startParallelMembership();
```

The cluster needs a committed rank assignment before workers start. The current low-level API exposes the steps directly:

```java
membership.voteForLeader();
membership.publishAssignmentAsLeader();
```

Production code should wait for all expected nodes to be visible before publishing the assignment.

## Start Workers

Workers build all ranks assigned to the local physical node and publish HTTP rank endpoints into gossip.

```java
HttpTensorParallelCollectiveServer collectiveServer = new HttpTensorParallelCollectiveServer(
        new InetSocketAddress("127.0.0.1", 0), Duration.ofSeconds(30));
collectiveServer.start();

Function<TensorParallelContext, TensorParallelCollectives> collectivesFactory =
        context -> new HttpTensorParallelCollectives(context, collectiveServer.uri());

TensorParallelWorker worker = TensorParallelWorker.start(
        nodeBuilder,
        membership,
        collectivesFactory,
        "127.0.0.1");
```

Each worker starts one rank server per local assigned rank.

## Run Generate

The coordinator discovers all rank endpoints, builds a `TensorParallelGenerationGroup`, and calls `generate(...)`.

```java
List<TensorParallelRankEndpoint> endpoints = membership.rankEndpointsForAssignment();

TensorParallelGenerationGroup group = TensorParallelGenerationGroup.fromEndpoints(endpoints.stream()
        .map(endpoint -> new TensorParallelGenerationGroup.RankEndpoint(
                endpoint.rank(),
                deploymentSpec.requestedNodes(),
                new HttpTensorParallelRankClient(URI.create(endpoint.uri())),
                false))
        .toList());

try (group;
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
and response construction. Transformer prefill and decode execution goes through the rank endpoints.

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

The test proves the full path: two gossip nodes, two workers, four HTTP rank endpoints, HTTP collectives, and
`TensorParallelGenerationGroup.generate(...)` producing a `Response`.

## Operational Notes

* Keep all nodes on the same model files and Deliverance build.
* Do not mix model families in one deployment.
* Use one deployment id per logical model deployment.
* Wait for all rank endpoints before serving traffic.
* Treat non-Gemma2 model families as unsupported until they have explicit tensor-parallel tests.
