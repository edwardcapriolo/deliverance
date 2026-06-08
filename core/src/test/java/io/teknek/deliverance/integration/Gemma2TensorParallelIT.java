package io.teknek.deliverance.integration;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.model.tensorparallel.GossipParallelMembership;
import io.teknek.deliverance.model.tensorparallel.GossipParallelSettings;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.TensorParallelDeploymentSpec;
import io.teknek.deliverance.model.tensorparallel.TensorParallelGenerationGroup;
import io.teknek.deliverance.model.tensorparallel.TensorParallelRankEndpoint;
import io.teknek.deliverance.model.tensorparallel.TensorParallelWorker;
import io.teknek.deliverance.model.tensorparallel.transport.HttpTensorParallelCollectiveServer;
import io.teknek.deliverance.model.tensorparallel.transport.HttpTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.transport.HttpTensorParallelRankClient;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.gossip.GossipSettings;
import io.teknek.gossip.Member;
import io.teknek.gossip.RemoteMember;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.net.InetSocketAddress;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.UUID;
import java.util.function.BooleanSupplier;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class Gemma2TensorParallelIT {

    @Test
    public void assignedGemma2RanksGenerateThroughTwoWorkers() throws Exception {
        String cluster = "deliverance-gemma2-tp-" + UUID.randomUUID();
        int basePort = 42_000 + Math.floorMod(cluster.hashCode(), 1_000);
        URI node0Uri = new URI("udp://127.0.0.1:" + basePort);
        URI node1Uri = new URI("udp://127.0.0.1:" + (basePort + 1));
        List<Member> seedMembers = List.of(
                new RemoteMember(cluster, node0Uri, "node-0"),
                new RemoteMember(cluster, node1Uri, "node-1")
        );
        GossipSettings settings = new GossipSettings();
        settings.setPersistRingState(false);
        settings.setPersistDataState(false);
        settings.setGossipInterval(100);
        settings.setCleanupInterval(2_000);

        ModelFetcher fetcher = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
        TensorParallelDeploymentSpec deploymentSpec = new TensorParallelDeploymentSpec("demo", 4, 2);
        MetricRegistry node0Metrics = new MetricRegistry();
        MetricRegistry node1Metrics = new MetricRegistry();
        WrappedForkJoinPool node0Pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        WrappedForkJoinPool node1Pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        TensorAllocator node0Allocator = new ArrayQueueTensorAllocator(node0Metrics);
        TensorAllocator node1Allocator = new ArrayQueueTensorAllocator(node1Metrics);
        AutoModelForCausaLm.Builder node0Builder = AutoModelForCausaLm.newBuilder(fetcher)
                .withMetricRegistry(node0Metrics)
                .withWrappedForkJoinPool(node0Pool)
                .withTensorAllocator(node0Allocator)
                .withTensorProvider(panamaProvider(node0Allocator, node0Pool))
                .withParallelSettings(new GossipParallelSettings(cluster, "node-0", node0Uri, seedMembers, settings,
                        deploymentSpec));
        AutoModelForCausaLm.Builder node1Builder = AutoModelForCausaLm.newBuilder(fetcher)
                .withMetricRegistry(node1Metrics)
                .withWrappedForkJoinPool(node1Pool)
                .withTensorAllocator(node1Allocator)
                .withTensorProvider(panamaProvider(node1Allocator, node1Pool))
                .withParallelSettings(new GossipParallelSettings(cluster, "node-1", node1Uri, seedMembers, settings,
                        deploymentSpec));

        try (node0Pool;
             node1Pool;
             GossipParallelMembership node0 = node0Builder.startParallelMembership();
             GossipParallelMembership node1 = node1Builder.startParallelMembership()) {
            eventually(() -> node0.liveMembers().size() == 1 && node1.liveMembers().size() == 1, Duration.ofSeconds(10));
            eventually(() -> node0.candidateNodeIds().size() == 2 && node1.candidateNodeIds().size() == 2,
                    Duration.ofSeconds(10));
            node0.voteForLeader();
            node1.voteForLeader();
            eventually(() -> "node-0".equals(node0.electedLeader()) && "node-0".equals(node1.electedLeader()),
                    Duration.ofSeconds(10));
            node0.publishAssignmentAsLeader();
            eventually(() -> node0.findAssignment() != null && node1.findAssignment() != null, Duration.ofSeconds(10));

            try (HttpTensorParallelCollectiveServer collectiveServer = new HttpTensorParallelCollectiveServer(
                    new InetSocketAddress("127.0.0.1", 0), Duration.ofSeconds(30))) {
                collectiveServer.start();
                Function<TensorParallelContext, TensorParallelCollectives> collectivesFactory =
                        context -> new HttpTensorParallelCollectives(context, collectiveServer.uri());

            try (TensorParallelWorker worker0 = TensorParallelWorker.start(node0Builder, node0, collectivesFactory, "127.0.0.1");
                 TensorParallelWorker worker1 = TensorParallelWorker.start(node1Builder, node1, collectivesFactory, "127.0.0.1")) {
                eventually(() -> node0.findRankEndpoints("node-0").size() == 2
                        && node0.findRankEndpoints("node-1").size() == 2
                        && node1.findRankEndpoints("node-0").size() == 2
                        && node1.findRankEndpoints("node-1").size() == 2, Duration.ofSeconds(10));

                List<TensorParallelRankEndpoint> endpoints = new ArrayList<>();
                endpoints.addAll(node0.findRankEndpoints("node-0"));
                endpoints.addAll(node0.findRankEndpoints("node-1"));
                endpoints.sort(Comparator.comparingInt(TensorParallelRankEndpoint::rank));

                TensorParallelGenerationGroup group = TensorParallelGenerationGroup.fromEndpoints(endpoints.stream()
                        .map(endpoint -> new TensorParallelGenerationGroup.RankEndpoint(endpoint.rank(), 4,
                                new HttpTensorParallelRankClient(URI.create(endpoint.uri())), false))
                        .toList());

                MetricRegistry coordinatorMetrics = new MetricRegistry();
                WrappedForkJoinPool coordinatorPool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
                TensorAllocator coordinatorAllocator = new ArrayQueueTensorAllocator(coordinatorMetrics);
                try (coordinatorPool;
                     group;
                     AbstractModel coordinatorModel = AutoModelForCausaLm.newBuilder(fetcher)
                             .withMetricRegistry(coordinatorMetrics)
                             .withWrappedForkJoinPool(coordinatorPool)
                             .withTensorAllocator(coordinatorAllocator)
                             .withTensorProvider(panamaProvider(coordinatorAllocator, coordinatorPool))
                             .build()) {
                    {
                        var prompt = coordinatorModel.promptSupport().get().builder()
                                .addUserMessage("What is 1 + 1?")
                                .build();
                        Response response = group.generate(coordinatorModel,
                                prompt,
                                new GeneratorParameters()
                                        .withNtokens(64)
                                        .withMaxTokens(1)
                                        .withTemperature(0.0f)
                                        .withSeed(123),
                                new DoNothingGenerateEvent());

                        assertNotNull(response);
                        assertEquals(1, response.generatedTokens.size());
                        assertEquals(1, response.samplerReturns.size());
                    }

                    {
                        var prompt = coordinatorModel.promptSupport().get().builder()
                                .addUserMessage("What is tensor parallelism?")
                                .build();
                        //assertSingleModelAndTensorParallelFirstTokenMatch(coordinatorModel, group, prompt.getPrompt());
                        Response response = group.generate(coordinatorModel,
                                prompt,
                                new GeneratorParameters()
                                        .withNtokens(64)
                                        .withMaxTokens(50)
                                        .withTemperature(0.0f)
                                        .withSeed(123),
                                new GenerateEvent() {
                                    @Override
                                    public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                                        System.out.println(nextRaw);
                                    }
                                });

                       System.out.println(response);
                    }

                }


            }
            }
        }
    }

    private static void eventually(BooleanSupplier condition, Duration timeout) throws InterruptedException {
        long deadline = System.nanoTime() + timeout.toNanos();
        while (System.nanoTime() < deadline) {
            if (condition.getAsBoolean()) {
                return;
            }
            Thread.sleep(100);
        }
        throw new AssertionError("condition did not become true within " + timeout);
    }

    private static void assertSingleModelAndTensorParallelFirstTokenMatch(AbstractModel singleModel,
            TensorParallelGenerationGroup group, String renderedPrompt) {
        int[] promptTokens = constructPromptTokensLikeGenerate(singleModel, renderedPrompt);
        float maxAbsDiff;
        try (AbstractTensor singlePrefill = singleModel.batchForward(promptTokens, 0);
             AbstractTensor tpPrefill = group.batchForward(promptTokens, 0)) {
            maxAbsDiff = maxAbsDiff(singlePrefill, tpPrefill);
        }

        var prompt = singleModel.promptSupport().get().builder()
                .addUserMessage("What is tensor parallelism?")
                .build();
        GeneratorParameters params = new GeneratorParameters()
                .withNtokens(64)
                .withMaxTokens(1)
                .withTemperature(0.0f)
                .withSeed(123);
        Response single = singleModel.generate(UUID.randomUUID(), prompt, params, new DoNothingGenerateEvent());
        Response tp = group.generate(singleModel, prompt, new GeneratorParameters()
                        .withNtokens(64)
                        .withMaxTokens(1)
                        .withTemperature(0.0f)
                        .withSeed(123),
                new DoNothingGenerateEvent());

        int singleToken = single.generatedTokens.get(0);
        int tpToken = tp.generatedTokens.get(0);
        assertEquals(singleToken, tpToken,
                "single-vs-TP first token mismatch; promptTokens=" + promptTokens.length
                        + " prefillMaxAbsDiff=" + maxAbsDiff
                        + " singleText='" + single.responseText + "'"
                        + " tpText='" + tp.responseText + "'");
    }

    private static float maxAbsDiff(AbstractTensor expected, AbstractTensor actual) {
        assertEquals(expected.shape().first(), actual.shape().first(), "prefill row count");
        assertEquals(expected.shape().last(), actual.shape().last(), "prefill column count");
        float max = 0.0f;
        for (int row = 0; row < expected.shape().first(); row++) {
            for (int col = 0; col < expected.shape().last(); col++) {
                max = Math.max(max, Math.abs(expected.get(row, col) - actual.get(row, col)));
            }
        }
        return max;
    }

    private static int[] constructPromptTokensLikeGenerate(AbstractModel model, String renderedPrompt) {
        long[] encoded = model.encodeForRuntime(renderedPrompt);
        if (encoded.length > 0 && encoded[0] == model.getConfig().bosToken) {
            encoded = Arrays.copyOfRange(encoded, 1, encoded.length);
        }
        int[] promptTokens = new int[encoded.length + 1];
        promptTokens[0] = model.getConfig().bosToken;
        for (int i = 0; i < encoded.length; i++) {
            promptTokens[i + 1] = Math.toIntExact(encoded[i]);
        }
        return promptTokens;
    }

    private static ConfigurableTensorProvider panamaProvider(TensorAllocator allocator, WrappedForkJoinPool pool) {
        return new ConfigurableTensorProvider(new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool));
    }

}
