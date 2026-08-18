package io.teknek.deliverance.integration.tensorparallel;

import com.codahale.metrics.MetricRegistry;
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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.net.URI;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.function.BooleanSupplier;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertNotNull;

public class TensorParallelGenerationIT {
    private static final String NODE_0 = "node-0";
    private static final String NODE_1 = "node-1";

    @ParameterizedTest
    @Tag("longtest")
    @MethodSource("modelCases")
    public void assignedRanksGenerateThroughTwoWorkers(ModelCase modelCase) throws Exception {
        String cluster = "deliverance-tp-" + UUID.randomUUID();
        int basePort = 42_000 + Math.floorMod(cluster.hashCode(), 1_000);
        URI node0Uri = new URI("udp://127.0.0.1:" + basePort);
        URI node1Uri = new URI("udp://127.0.0.1:" + (basePort + 1));
        List<Member> seedMembers = List.of(new RemoteMember(cluster, node0Uri, NODE_0),
                new RemoteMember(cluster, node1Uri, NODE_1));
        GossipSettings settings = new GossipSettings();
        settings.setPersistRingState(false);
        settings.setPersistDataState(false);
        settings.setGossipInterval(100);
        settings.setCleanupInterval(2_000);

        ModelFetcher fetcher = new ModelFetcher(modelCase.owner(), modelCase.modelName());
        TensorParallelDeploymentSpec deploymentSpec = new TensorParallelDeploymentSpec("demo",
                modelCase.tensorParallelSize(), modelCase.maxRanksPerWorker());
        try (TestNode node0 = createNode(fetcher, cluster, NODE_0, node0Uri, seedMembers, settings, deploymentSpec);
             TestNode node1 = createNode(fetcher, cluster, NODE_1, node1Uri, seedMembers, settings, deploymentSpec)) {
            List<TestNode> nodes = List.of(node0, node1);
            eventually(() -> allMembersVisible(nodes), Duration.ofSeconds(10));
            eventually(() -> allCandidatesVisible(nodes, deploymentSpec.minimumPhysicalNodes()), Duration.ofSeconds(10));
            eventually(() -> allNodesSeeLeader(nodes, NODE_0), Duration.ofSeconds(10));
            eventually(() -> allNodesSeeAssignment(nodes), Duration.ofSeconds(10));
            eventually(() -> allNodesSeeCollectiveUri(nodes), Duration.ofSeconds(10));
            eventually(() -> allNodesSeeRankEndpoints(nodes), Duration.ofSeconds(60));

            TensorParallelGenerationGroup group = node0.membership().openGenerationGroup();

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
                            .buildLocalTransformerModel()) {
                    var prompt = coordinatorModel.promptSupport().get().builder()
                            .addUserMessage("What is the capital of New York, USA?")
                            .build();
                    long tpStart = System.nanoTime();
                    Response response = group.generate(coordinatorModel, prompt, new GeneratorParameters()
                                    .withNtokens(64)
                                    .withMaxTokens(3)
                                    .withTemperature(0.0f)
                                    .withSeed(123),
                            new DoNothingGenerateEvent());
                    long tpMillis = (System.nanoTime() - tpStart) / 1_000_000L;

                    System.out.printf("TP_RESULT model=%s/%s tp_ms=%d tokens=%s text=%s special=%s%n",
                            modelCase.owner(), modelCase.modelName(), tpMillis, response.generatedTokens,
                            response.responseText, response.responseTextWithSpecialTokens);
                    assertNotNull(response);
                }

        }
    }

    private static Stream<ModelCase> modelCases() {
        return Stream.of(
                new ModelCase("tjake", "gemma-2-2b-it-JQ4", 4, 2)//,
                //new ModelCase("edwardcapriolo", "Qwen3-4B-JQ4", 4, 2)
        );
    }

    private static TestNode createNode(ModelFetcher fetcher, String cluster, String nodeId, URI nodeUri,
            List<Member> seedMembers, GossipSettings settings, TensorParallelDeploymentSpec deploymentSpec) {
        MetricRegistry metrics = new MetricRegistry();
        WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher)
                .withMetricRegistry(metrics)
                .withWrappedForkJoinPool(pool)
                .withTensorAllocator(allocator)
                .withTensorProvider(panamaProvider(allocator, pool))

                .withParallelSettings(new GossipParallelSettings(cluster, nodeId, nodeUri, seedMembers, settings,
                        deploymentSpec, "netty"))
                .buildAbstractModel();
        return new TestNode(nodeId, model, pool, model.gossipParallelMembership().orElseThrow());
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
        return nodes.stream().allMatch(observer -> nodes.stream()
                .allMatch(owner -> observer.membership().findRankEndpoints(owner.id()).size() == 2));
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

    private record TestNode(String id, AbstractModel model, WrappedForkJoinPool pool,
            GossipParallelMembership membership) implements AutoCloseable {
        @Override
        public void close() {
            model.close();
            pool.close();
        }
    }

    private record ModelCase(String owner, String modelName, int tensorParallelSize, int maxRanksPerWorker) {
        @Override
        public String toString() {
            return owner + "/" + modelName + " tp=" + tensorParallelSize + " ranksPerWorker=" + maxRanksPerWorker;
        }
    }

}
