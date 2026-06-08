package io.teknek.deliverance.model.tensorparallel;

import io.teknek.gossip.GossipSettings;
import io.teknek.gossip.Member;
import io.teknek.gossip.RemoteMember;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.gemma2.Gemma2Model;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

public class GossipTensorParallelMembershipTest {

    @Test
    public void twoNodesJoinAndShareTensorParallelDeploymentMetadata() throws Exception {
        String cluster = "deliverance-tp-" + UUID.randomUUID();
        int basePort = 41_000 + Math.floorMod(cluster.hashCode(), 1_000);
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
        int tensorParallelRanks = assertGemma2TensorParallelRankCapacity(fetcher, 10);
        TensorParallelDeploymentSpec deploymentSpec = new TensorParallelDeploymentSpec("demo", "gemma2", tensorParallelRanks, 2);

        AutoModelForCausaLm.Builder node0Builder = AutoModelForCausaLm.newBuilder(fetcher)
                .withParallelSettings(new GossipParallelSettings(cluster, "node-0", node0Uri, seedMembers, settings,
                        deploymentSpec));
        AutoModelForCausaLm.Builder node1Builder = AutoModelForCausaLm.newBuilder(fetcher)
                .withParallelSettings(new GossipParallelSettings(cluster, "node-1", node1Uri, seedMembers, settings,
                        deploymentSpec));

        try (GossipParallelMembership node0 = node0Builder.startParallelMembership();
             GossipParallelMembership node1 = node1Builder
                     .startParallelMembership()) {

            eventually(() -> node0.liveMembers().size() == 1 && node1.liveMembers().size() == 1,
                    Duration.ofSeconds(10));

            node0.publishDeploymentSpec();

            eventually(() -> deploymentSpec.equals(node1.findDeploymentSpec()),
                    Duration.ofSeconds(10));
            eventually(() -> node0.candidateNodeIds().size() == 2 && node1.candidateNodeIds().size() == 2,
                    Duration.ofSeconds(10));

            TensorParallelTopology node0Topology = assertConvergedFourRankTopology(deploymentSpec, node0, node1);

            node0.voteForLeader();
            node1.voteForLeader();
            eventually(() -> "node-0".equals(node0.electedLeader()) && "node-0".equals(node1.electedLeader()),
                    Duration.ofSeconds(10));
            assertEquals("node-0", node0.electedLeader());
            assertEquals("node-0", node1.electedLeader());

            node0.publishAssignmentAsLeader();
            eventually(() -> node0.findAssignment() != null && node1.findAssignment() != null,
                    Duration.ofSeconds(10));

            TensorParallelAssignment assignment = node1.findAssignment();
            assertEquals("demo", assignment.deploymentId());
            assertEquals("node-0", assignment.leaderNodeId());
            assertEquals(4, assignment.tensorParallelSize());
            assertEquals(node0Topology.assignmentHash(), assignment.assignmentHash());
            assertTrue(assignment.assignsEveryRank());
            assertTrue(assignment.matchesTopology(node0Topology));
            assertTrue(node0.assignmentMatchesLocalTopology());
            assertTrue(node1.assignmentMatchesLocalTopology());
            assertEquals(List.of(
                    new TensorParallelRankAssignment(0, "node-0"),
                    new TensorParallelRankAssignment(1, "node-0"),
                    new TensorParallelRankAssignment(2, "node-1"),
                    new TensorParallelRankAssignment(3, "node-1")
            ), assignment.ranks());
            assertEquals(List.of(0, 1), node0.localRanks());
            assertEquals(List.of(2, 3), node1.localRanks());

            List<AutoModelForCausaLm.Builder> node0RankBuilders = node0Builder.localAssignedRankBuilders(node0);
            List<AutoModelForCausaLm.Builder> node1RankBuilders = node1Builder.localAssignedRankBuilders(node1);
            assertEquals(List.of(0, 1), node0RankBuilders.stream()
                    .map(builder -> builder.getTensorParallelContext().rank()).toList());
            assertEquals(List.of(2, 3), node1RankBuilders.stream()
                    .map(builder -> builder.getTensorParallelContext().rank()).toList());
            assertEquals(4, node0RankBuilders.get(0).getTensorParallelContext().size());
            assertEquals(4, node1RankBuilders.get(0).getTensorParallelContext().size());
        }
    }

    private static int assertGemma2TensorParallelRankCapacity(ModelFetcher fetcher, int requestedNodes) {
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher).build()) {
            assertInstanceOf(Gemma2Model.class, model);
            assertEquals(8, model.getConfig().numberOfHeads);
            assertEquals(4, model.getConfig().numberOfKeyValueHeads);
            int tensorParallelRanks = TensorParallelPlanner.chooseSize(model.getConfig(), requestedNodes);
            assertEquals(4, tensorParallelRanks);
            return tensorParallelRanks;
        }
    }

    private static TensorParallelTopology assertConvergedFourRankTopology(TensorParallelDeploymentSpec deploymentSpec,
            GossipParallelMembership node0, GossipParallelMembership node1) {
        assertEquals(deploymentSpec, node1.findDeploymentSpec());
        TensorParallelTopology node0Topology = node0.topology();
        TensorParallelTopology node1Topology = node1.topology();
        assertEquals(node0Topology, node1Topology);
        assertEquals(4, node0Topology.availableSlots());
        assertEquals(4, node0Topology.tensorParallelSize());
        assertEquals(List.of("node-0", "node-0", "node-1", "node-1"), node0Topology.activeRankAssignments());
        assertEquals(List.of(), node0Topology.standbyNodeIds());
        assertEquals(0, node0Topology.rankOf("node-0"));
        assertEquals(2, node0Topology.rankOf("node-1"));
        assertEquals(2, node0Topology.rankCountFor("node-0"));
        assertEquals(2, node0Topology.rankCountFor("node-1"));
        assertEquals(List.of(0, 1), node0Topology.assignedRanks("node-0"));
        assertEquals(List.of(2, 3), node0Topology.assignedRanks("node-1"));
        return node0Topology;
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
}
