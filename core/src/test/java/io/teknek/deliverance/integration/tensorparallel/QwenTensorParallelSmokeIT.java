package io.teknek.deliverance.integration.tensorparallel;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.model.tensorparallel.GossipParallelMembership;
import io.teknek.deliverance.model.tensorparallel.GossipParallelSettings;
import io.teknek.deliverance.model.tensorparallel.TensorParallelDeploymentSpec;
import io.teknek.deliverance.model.tensorparallel.TensorParallelGenerationGroup;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.gossip.GossipSettings;
import io.teknek.gossip.Member;
import io.teknek.gossip.RemoteMember;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.time.Duration;
import java.util.List;
import java.util.Locale;
import java.util.UUID;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class QwenTensorParallelSmokeIT {
    private static final String NODE_0 = "node-0";
    private static final String NODE_1 = "node-1";

    @Test
    @Tag("longtest")
    public void qwen06bTwoWorkersOneCoordinatorSimpleChatIsSane() throws Exception {
        runQwen06bSmoke(false, true);
    }

    @Test
    @Tag("longtest")
    public void qwen06bDefaultNativeTpProviderTwoWorkersOneCoordinatorSimpleChatIsSane() throws Exception {
        runQwen06bSmoke(true, false);
    }

    @Test
    @Tag("longtest")
    public void qwen06bTwoWorkersPromptSurvivesWorkerRestart() throws Exception {
        String cluster = "deliverance-qwen-tp-restart-" + UUID.randomUUID();
        int basePort = 44_000 + Math.floorMod(cluster.hashCode(), 1_000);
        URI node0Uri = new URI("udp://127.0.0.1:" + basePort);
        URI node1Uri = new URI("udp://127.0.0.1:" + (basePort + 1));
        URI node1RestartUri = new URI("udp://127.0.0.1:" + (basePort + 11));
        List<Member> seedMembers = List.of(new RemoteMember(cluster, node0Uri, NODE_0),
                new RemoteMember(cluster, node1Uri, NODE_1), new RemoteMember(cluster, node1RestartUri, NODE_1));
        GossipSettings settings = gossipSettings();
        ModelFetcher fetcher = new ModelFetcher("edwardcapriolo", "Qwen3-0.6B-JQ4");
        TensorParallelDeploymentSpec deploymentSpec = new TensorParallelDeploymentSpec("qwen-restart", 2, 1);

        try (TestNode node0 = createNode(fetcher, cluster, NODE_0, node0Uri, seedMembers, settings, deploymentSpec,
                true);
             TestNode node1 = createNode(fetcher, cluster, NODE_1, node1Uri, seedMembers, settings, deploymentSpec,
                     true)) {
            waitForReady(List.of(node0, node1), deploymentSpec);
            try (Coordinator coordinator = coordinator(fetcher, node0.membership(), true)) {
                assertPrompt(coordinator);
                String oldNode1Endpoint = node0.membership().findRankEndpoints(NODE_1).stream()
                        .findFirst()
                        .orElseThrow(() -> new AssertionError("node-1 endpoint was not visible before restart"))
                        .uri();

                node1.close();
                eventually(() -> portReleased(basePort + 1), Duration.ofSeconds(10));

                try (TestNode restartedNode1 = createNode(fetcher, cluster, NODE_1, node1RestartUri, seedMembers,
                        settings, deploymentSpec, true)) {
                    eventually(() -> {
                        return node0.membership().findRankEndpoints(NODE_1).stream()
                                .anyMatch(endpoint -> !endpoint.uri().equals(oldNode1Endpoint));
                    }, Duration.ofSeconds(60));
                    eventually(() -> {
                        try {
                            return node0.membership().rankEndpointsForAssignment().stream()
                                    .noneMatch(endpoint -> endpoint.uri().equals(oldNode1Endpoint));
                        } catch (RuntimeException e) {
                            return false;
                        }
                    }, Duration.ofSeconds(60));
                    try (TensorParallelGenerationGroup recoveredGroup = node0.membership().openGenerationGroup()) {
                        assertPrompt(new Coordinator(coordinator.coordinatorModel(), null, recoveredGroup));
                    }
                }
            }
        }
    }

    @Test
    @Tag("longtest")
    public void qwen06bDefaultNativeProviderSingleModelSimpleChatIsSane() {
        ModelFetcher fetcher = new ModelFetcher("edwardcapriolo", "Qwen3-0.6B-JQ4");
        MetricRegistry metrics = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
             AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher)
                     .withMetricRegistry(metrics)
                     .withWrappedForkJoinPool(pool)
                     .withTensorAllocator(allocator)
                     .buildLocalTransformerModel()) {
            var prompt = model.promptSupport().get().builder()
                    .addTemplateArg("enable_thinking", true)
                    .addUserMessage("hi")
                    .build();
            Response response = model.generate(UUID.randomUUID(), prompt, new GeneratorParameters()
                            .withNtokens(128)
                            .withMaxTokens(64)
                            .withTemperature(0.0f)
                            .withSeed(123),
                    new DoNothingGenerateEvent());
            System.out.printf("QWEN_NATIVE_SINGLE_SMOKE tokens=%s text=%s special=%s%n", response.generatedTokens,
                    response.responseText, response.responseTextWithSpecialTokens);
            assertSaneGreeting(response.responseTextWithSpecialTokens);
        }
    }

    private void runQwen06bSmoke(boolean defaultProvider, boolean exercisePrefixCache) throws Exception {
        String cluster = "deliverance-qwen-tp-smoke-" + UUID.randomUUID();
        int basePort = 43_000 + Math.floorMod(cluster.hashCode(), 1_000);
        URI node0Uri = new URI("udp://127.0.0.1:" + basePort);
        URI node1Uri = new URI("udp://127.0.0.1:" + (basePort + 1));
        List<Member> seedMembers = List.of(new RemoteMember(cluster, node0Uri, NODE_0),
                new RemoteMember(cluster, node1Uri, NODE_1));
        GossipSettings settings = gossipSettings();

        ModelFetcher fetcher = new ModelFetcher("edwardcapriolo", "Qwen3-0.6B-JQ4");
        TensorParallelDeploymentSpec deploymentSpec = new TensorParallelDeploymentSpec("qwen-smoke", 4, 2);
        try (TestNode node0 = createNode(fetcher, cluster, NODE_0, node0Uri, seedMembers, settings, deploymentSpec,
                defaultProvider);
            TestNode node1 = createNode(fetcher, cluster, NODE_1, node1Uri, seedMembers, settings, deploymentSpec,
                      defaultProvider)) {
            List<TestNode> nodes = List.of(node0, node1);
            waitForReady(nodes, deploymentSpec);

            TensorParallelGenerationGroup group = node0.membership().openGenerationGroup();
            MetricRegistry coordinatorMetrics = new MetricRegistry();
            WrappedForkJoinPool coordinatorPool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
            TensorAllocator coordinatorAllocator = new ArrayQueueTensorAllocator(coordinatorMetrics);
            try (coordinatorPool;
                 group;
                 AbstractModel coordinatorModel = coordinatorBuilder(fetcher, coordinatorMetrics, coordinatorPool,
                         coordinatorAllocator, defaultProvider).buildLocalTransformerModel()) {
                var prompt = coordinatorModel.promptSupport().get().builder()
                        .addTemplateArg("enable_thinking", true)
                        .addUserMessage("hi")
                        .build();
                GeneratorParameters parameters = new GeneratorParameters()
                                .withNtokens(128)
                                .withMaxTokens(64)
                                .withTemperature(0.0f)
                                .withSeed(123);
                Response legacyForwarderResponse = coordinatorModel.generateWithForwarder(UUID.randomUUID(), prompt,
                        parameters,
                        new DoNothingGenerateEvent(),
                        new AbstractModel.GenerationForwarder() {
                            @Override
                            public io.teknek.deliverance.tensor.AbstractTensor batchForward(int[] tokenIds, int startPosition) {
                                return group.batchForward(tokenIds, startPosition);
                            }

                            @Override
                            public io.teknek.deliverance.tensor.AbstractTensor forward(int tokenId, int position) {
                                return group.forward(tokenId, position);
                            }
                        });
                System.out.printf("QWEN_TP_SMOKE_LEGACY tokens=%s text=%s special=%s%n",
                        legacyForwarderResponse.generatedTokens, legacyForwarderResponse.responseText,
                        legacyForwarderResponse.responseTextWithSpecialTokens);

                Response response = group.generate(UUID.randomUUID(), coordinatorModel, prompt, parameters,
                        new DoNothingGenerateEvent());

                System.out.printf("QWEN_TP_SMOKE tokens=%s text=%s special=%s%n", response.generatedTokens,
                        response.responseText, response.responseTextWithSpecialTokens);
                assertSaneGreeting(response.responseTextWithSpecialTokens);

                if (!exercisePrefixCache) {
                    return;
                }

                var longPrompt = coordinatorModel.promptSupport().get().builder()
                        .addTemplateArg("enable_thinking", true)
                        .addSystemMessage("You are a concise assistant. Keep replies short. This repeated prefix exists to exercise tensor parallel prefix cache storage and restore mechanics without using compressed KV snapshots.")
                        .addUserMessage("hi")
                        .build();
                GeneratorParameters shortParameters = new GeneratorParameters()
                        .withNtokens(160)
                        .withMaxTokens(24)
                        .withTemperature(0.0f)
                        .withSeed(123);
                Response coldLong = group.generate(UUID.randomUUID(), coordinatorModel, longPrompt, shortParameters,
                        new DoNothingGenerateEvent());
                System.out.printf("QWEN_TP_SMOKE_LONG_COLD tokens=%s text=%s special=%s%n", coldLong.generatedTokens,
                        coldLong.responseText, coldLong.responseTextWithSpecialTokens);
                assertSaneGreeting(coldLong.responseTextWithSpecialTokens);

                Response cachedLong = group.generate(UUID.randomUUID(), coordinatorModel, longPrompt, shortParameters,
                        new DoNothingGenerateEvent());
                System.out.printf("QWEN_TP_SMOKE_LONG_CACHED tokens=%s text=%s special=%s%n", cachedLong.generatedTokens,
                        cachedLong.responseText, cachedLong.responseTextWithSpecialTokens);
                assertSaneGreeting(cachedLong.responseTextWithSpecialTokens);
            }
        }
    }

    private static Coordinator coordinator(ModelFetcher fetcher, GossipParallelMembership membership,
            boolean defaultProvider) {
        MetricRegistry coordinatorMetrics = new MetricRegistry();
        WrappedForkJoinPool coordinatorPool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        TensorAllocator coordinatorAllocator = new ArrayQueueTensorAllocator(coordinatorMetrics);
        AbstractModel coordinatorModel = coordinatorBuilder(fetcher, coordinatorMetrics, coordinatorPool,
                coordinatorAllocator, defaultProvider).buildLocalTransformerModel();
        TensorParallelGenerationGroup group = membership.openGenerationGroup();
        return new Coordinator(coordinatorModel, coordinatorPool, group);
    }

    private static void assertPrompt(Coordinator coordinator) {
        var prompt = coordinator.coordinatorModel().promptSupport().get().builder()
                .addTemplateArg("enable_thinking", true)
                .addUserMessage("hi")
                .build();
        Response response = coordinator.group().generate(UUID.randomUUID(), coordinator.coordinatorModel(), prompt, new GeneratorParameters()
                        .withNtokens(64)
                        .withMaxTokens(16)
                        .withTemperature(0.0f)
                        .withSeed(123),
                new DoNothingGenerateEvent());
        assertNotNull(response);
        assertFalse(response.generatedTokens.isEmpty());
    }

    private static void waitForReady(List<TestNode> nodes, TensorParallelDeploymentSpec deploymentSpec) throws Exception {
        GossipParallelMembership observer = nodes.getFirst().membership();
        eventually(() -> observer.findAssignment() != null, Duration.ofSeconds(60));
        eventually(() -> observer.findCollectiveUri() != null, Duration.ofSeconds(60));
        eventually(() -> {
            try {
                return observer.rankEndpointsForAssignment().size() == deploymentSpec.requestedNodes();
            } catch (RuntimeException e) {
                return false;
            }
        }, Duration.ofSeconds(120));
    }

    private static boolean portReleased(int port) {
        try (java.net.ServerSocket ignored = new java.net.ServerSocket(port)) {
            return true;
        } catch (java.io.IOException e) {
            return false;
        }
    }

    private static AutoModelForCausaLm.Builder coordinatorBuilder(ModelFetcher fetcher, MetricRegistry metrics,
            WrappedForkJoinPool pool, TensorAllocator allocator, boolean defaultProvider) {
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(fetcher)
                .withMetricRegistry(metrics)
                .withWrappedForkJoinPool(pool)
                .withTensorAllocator(allocator);
        if (!defaultProvider) {
            builder.withTensorProvider(panamaProvider(allocator, pool));
        }
        return builder;
    }

    private static void assertSaneGreeting(String text) {
        String lower = text.toLowerCase(Locale.ROOT);
        assertFalse(lower.contains("weeds weeds weeds"), text);
        assertFalse(text.contains("ованияованияования"), text);
        assertTrue(lower.contains("hello") || lower.contains("hi!") || lower.contains("hi there") || lower.contains("assist")
                || lower.contains("help"), text);
    }

    private static TestNode createNode(ModelFetcher fetcher, String cluster, String nodeId, URI nodeUri,
            List<Member> seedMembers, GossipSettings settings, TensorParallelDeploymentSpec deploymentSpec,
            boolean defaultProvider) {
        MetricRegistry metrics = new MetricRegistry();
        WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(fetcher)
                .withMetricRegistry(metrics)
                .withWrappedForkJoinPool(pool)
                .withTensorAllocator(allocator)
                .withParallelSettings(new GossipParallelSettings(cluster, nodeId, nodeUri, seedMembers, settings,
                        deploymentSpec, "netty"));
        if (!defaultProvider) {
            builder.withTensorProvider(panamaProvider(allocator, pool));
        }
        AbstractModel model = builder.buildAbstractModel();
        return new TestNode(model, pool, model.gossipParallelMembership().orElseThrow());
    }

    private static GossipSettings gossipSettings() {
        GossipSettings settings = new GossipSettings();
        settings.setPersistRingState(false);
        settings.setPersistDataState(false);
        settings.setGossipInterval(100);
        settings.setCleanupInterval(2_000);
        return settings;
    }

    private static boolean allMembersVisible(List<TestNode> nodes) {
        return nodes.stream().allMatch(node -> node.membership().liveMembers().size() == nodes.size() - 1);
    }

    private static boolean allCandidatesVisible(List<TestNode> nodes, int expectedCandidates) {
        return nodes.stream().allMatch(node -> node.membership().candidateNodeIds().size() == expectedCandidates);
    }

    private static boolean allNodesSeeLeader(List<TestNode> nodes, String leaderNodeId) {
        return nodes.stream().allMatch(node -> leaderNodeId.equals(node.membership().electedLeader()));
    }

    private static boolean allNodesSeeAssignment(List<TestNode> nodes) {
        return nodes.stream().allMatch(node -> node.membership().findAssignment() != null);
    }

    private static boolean allNodesSeeCollectiveUri(List<TestNode> nodes) {
        return nodes.stream().allMatch(node -> node.membership().findCollectiveUri() != null);
    }

    private static boolean allNodesSeeRankEndpoints(List<TestNode> nodes) {
        int expectedRanks = nodes.getFirst().membership().requireAssignment().tensorParallelSize();
        return nodes.stream().allMatch(observer -> {
            try {
                return observer.membership().rankEndpointsForAssignment().size() == expectedRanks;
            } catch (RuntimeException e) {
                return false;
            }
        });
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

    private static ConfigurableTensorProvider panamaProvider(TensorAllocator allocator, WrappedForkJoinPool pool) {
        return new ConfigurableTensorProvider(new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool));
    }

    private record TestNode(AbstractModel model, WrappedForkJoinPool pool,
            GossipParallelMembership membership) implements AutoCloseable {
        @Override
        public void close() {
            model.close();
            pool.close();
        }
    }

    private record Coordinator(AbstractModel coordinatorModel, WrappedForkJoinPool pool, TensorParallelGenerationGroup group) implements AutoCloseable {
        @Override
        public void close() {
            group.close();
            if (pool != null) {
                coordinatorModel.close();
                pool.close();
            }
        }
    }
}
