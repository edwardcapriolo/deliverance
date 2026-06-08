package io.teknek.deliverance.integration;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.tensorparallel.GossipParallelMembership;
import io.teknek.deliverance.model.tensorparallel.GossipParallelSettings;
import io.teknek.deliverance.model.tensorparallel.InProcessTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.TensorParallelDeploymentSpec;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.gossip.GossipSettings;
import io.teknek.gossip.Member;
import io.teknek.gossip.RemoteMember;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.BooleanSupplier;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class Gemma2TensorParallelIT {

    @Test
    public void assignedGemma2RanksRunOnePrefillForward() throws Exception {
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
        TensorParallelDeploymentSpec deploymentSpec = new TensorParallelDeploymentSpec("demo", "gemma2", 4, 2);
        AutoModelForCausaLm.Builder node0Builder = AutoModelForCausaLm.newBuilder(fetcher)
                .withParallelSettings(new GossipParallelSettings(cluster, "node-0", node0Uri, seedMembers, settings,
                        deploymentSpec));
        AutoModelForCausaLm.Builder node1Builder = AutoModelForCausaLm.newBuilder(fetcher)
                .withParallelSettings(new GossipParallelSettings(cluster, "node-1", node1Uri, seedMembers, settings,
                        deploymentSpec));

        try (GossipParallelMembership node0 = node0Builder.startParallelMembership();
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

            InProcessTensorParallelCollectives.Group collectivesGroup = new InProcessTensorParallelCollectives.Group(
                    Duration.ofSeconds(30));
            Function<TensorParallelContext, TensorParallelCollectives> collectivesFactory =
                    context -> new InProcessTensorParallelCollectives(context, collectivesGroup);

            List<AbstractModel> models = new ArrayList<>();
            models.addAll(node0Builder.buildLocalAssignedRanks(node0, collectivesFactory));
            models.addAll(node1Builder.buildLocalAssignedRanks(node1, collectivesFactory));
            try {
                int[] promptTokens = models.get(0).constructPromptTokensForRuntime("What is 1 + 1?");
                List<AbstractTensor> outputs = runForwardOnAllRanks(models, promptTokens);
                try {
                    String first = normalize(TensorDisplayUtil.pretty2dDisplayAll(outputs.get(0)));
                    for (AbstractTensor output : outputs) {
                        assertEquals(first, normalize(TensorDisplayUtil.pretty2dDisplayAll(output)));
                    }
                } finally {
                    for (AbstractTensor output : outputs) {
                        output.close();
                    }
                }
            } finally {
                for (AbstractModel model : models) {
                    model.close();
                }
            }
        }
    }

    private static List<AbstractTensor> runForwardOnAllRanks(List<AbstractModel> models, int[] promptTokens) throws Exception {
        try (ExecutorService executor = Executors.newFixedThreadPool(models.size())) {
            List<Future<AbstractTensor>> futures = new ArrayList<>();
            for (AbstractModel model : models) {
                futures.add(executor.submit(() -> model.batchForward(promptTokens, 0)));
            }
            List<AbstractTensor> outputs = new ArrayList<>();
            for (Future<AbstractTensor> future : futures) {
                outputs.add(future.get());
            }
            return outputs;
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

    private static String normalize(String display) {
        return display.strip().replaceAll("(?m) +$", "");
    }
}
