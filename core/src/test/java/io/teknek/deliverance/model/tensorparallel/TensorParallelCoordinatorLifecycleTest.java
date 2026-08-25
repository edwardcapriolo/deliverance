package io.teknek.deliverance.model.tensorparallel;

import io.teknek.gossip.GossipSettings;
import io.teknek.gossip.Member;
import io.teknek.gossip.RemoteMember;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.URI;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class TensorParallelCoordinatorLifecycleTest {

    @Test
    void workersStartedBeforeCoordinatorAreDiscoveredAndCoordinatorCanRestart() throws Exception {
        String cluster = "tp-lifecycle-" + UUID.randomUUID();
        int coordinatorPort = freePort();
        int worker0Port = freePort();
        int worker1Port = freePort();
        URI coordinatorUri = URI.create("udp://127.0.0.1:" + coordinatorPort);
        URI worker0Uri = URI.create("udp://127.0.0.1:" + worker0Port);
        URI worker1Uri = URI.create("udp://127.0.0.1:" + worker1Port);
        TensorParallelDeploymentSpec spec = new TensorParallelDeploymentSpec("lifecycle", 2, 1);
        GossipSettings settings = gossipSettings();
        List<Member> workerSeeds = List.of(new RemoteMember(cluster, coordinatorUri, "coordinator"));
        List<Member> coordinatorSeeds = List.of(new RemoteMember(cluster, coordinatorUri, "coordinator"));

        try (GossipParallelMembership worker0 = GossipParallelMembership.start(new GossipParallelSettings(cluster,
                "worker-0", worker0Uri, workerSeeds, settings, spec, "netty"));
             GossipParallelMembership worker1 = GossipParallelMembership.start(new GossipParallelSettings(cluster,
                      "worker-1", worker1Uri, workerSeeds, settings, spec, "netty"))) {

            try (GossipParallelMembership coordinator = GossipParallelMembership.startObserver(new GossipParallelSettings(
                    cluster, "coordinator", coordinatorUri, coordinatorSeeds, settings, spec, "netty"))) {
                assertCoordinatorSeesWorkersAndAssignment(coordinator);
            }

            eventually(() -> portReleased(coordinatorPort), Duration.ofSeconds(10));

            try (GossipParallelMembership restartedCoordinator = GossipParallelMembership.startObserver(
                    new GossipParallelSettings(cluster, "coordinator", coordinatorUri, coordinatorSeeds, settings, spec,
                            "netty"))) {
                assertCoordinatorSeesWorkersAndAssignment(restartedCoordinator);
            }
        }
    }

    @Test
    void coordinatorObservesWorkerRestartWithFreshRankEndpoint() throws Exception {
        String cluster = "tp-worker-restart-" + UUID.randomUUID();
        int coordinatorPort = freePort();
        int worker0Port = freePort();
        int worker1Port = freePort();
        int worker1RestartPort = freePort();
        URI coordinatorUri = URI.create("udp://127.0.0.1:" + coordinatorPort);
        URI worker0Uri = URI.create("udp://127.0.0.1:" + worker0Port);
        URI worker1Uri = URI.create("udp://127.0.0.1:" + worker1Port);
        URI worker1RestartUri = URI.create("udp://127.0.0.1:" + worker1RestartPort);
        TensorParallelDeploymentSpec spec = new TensorParallelDeploymentSpec("restart", 2, 1);
        GossipSettings settings = gossipSettings();
        List<Member> workerSeeds = List.of(new RemoteMember(cluster, coordinatorUri, "coordinator"));
        List<Member> coordinatorSeeds = List.of(new RemoteMember(cluster, coordinatorUri, "coordinator"));

        try (GossipParallelMembership coordinator = GossipParallelMembership.startObserver(new GossipParallelSettings(
                cluster, "coordinator", coordinatorUri, coordinatorSeeds, settings, spec, "netty"));
             GossipParallelMembership worker0 = GossipParallelMembership.start(new GossipParallelSettings(cluster,
                      "worker-0", worker0Uri, workerSeeds, settings, spec, "netty"))) {

            GossipParallelMembership worker1 = GossipParallelMembership.start(new GossipParallelSettings(cluster,
                    "worker-1", worker1Uri, workerSeeds, settings, spec, "netty"));
            try {
                assertCoordinatorSeesWorkersAndAssignment(coordinator);
                worker0.publishRankEndpoints(List.of(new TensorParallelRankEndpoint(0, "worker-0",
                        "http://127.0.0.1:50000")));
                worker1.publishRankEndpoints(List.of(new TensorParallelRankEndpoint(1, "worker-1",
                        "http://127.0.0.1:50001")));

                eventually(() -> coordinator.findRankEndpoints("worker-1").stream()
                                .anyMatch(endpoint -> endpoint.rank() == 1
                                        && endpoint.uri().equals("http://127.0.0.1:50001")),
                        Duration.ofSeconds(10));
            } finally {
                worker1.close();
            }

            eventually(() -> portReleased(worker1Port), Duration.ofSeconds(10));

            try (GossipParallelMembership restartedWorker1 = GossipParallelMembership.start(new GossipParallelSettings(cluster,
                    "worker-1", worker1RestartUri, workerSeeds, settings, spec, "netty"))) {
                restartedWorker1.publishRankEndpoints(List.of(new TensorParallelRankEndpoint(1, "worker-1",
                        "http://127.0.0.1:50101")));

                eventually(() -> coordinator.findRankEndpoints("worker-1").stream()
                                .anyMatch(endpoint -> endpoint.rank() == 1
                                        && endpoint.uri().equals("http://127.0.0.1:50101")),
                        Duration.ofSeconds(10));

                assertEquals("http://127.0.0.1:50101", coordinator.findRankEndpoints("worker-1").getFirst().uri());
            }
        }
    }

    @Disabled
            //s@Test
    void coordinatorRestartsWithNewUriAndRecoversThreeWorkers() throws Exception {
        String cluster = "tp-coordinator-restart-" + UUID.randomUUID();
        int coordinatorPort = freePort();
        int restartedCoordinatorPort = freePort();
        URI coordinatorUri = URI.create("udp://127.0.0.1:" + coordinatorPort);
        URI restartedCoordinatorUri = URI.create("udp://127.0.0.1:" + restartedCoordinatorPort);
        TensorParallelDeploymentSpec spec = new TensorParallelDeploymentSpec("coordinator-restart", 3, 1);
        GossipSettings settings = gossipSettings();
        List<Member> workerSeeds = List.of(new RemoteMember(cluster, coordinatorUri, "coordinator"),
                new RemoteMember(cluster, restartedCoordinatorUri, "coordinator"));
        List<Member> coordinatorSeeds = List.of(new RemoteMember(cluster, coordinatorUri, "coordinator"));
        List<Member> restartedCoordinatorSeeds = List.of(new RemoteMember(cluster, restartedCoordinatorUri, "coordinator"));

        try (GossipParallelMembership worker0 = GossipParallelMembership.start(new GossipParallelSettings(cluster,
                "worker-0", URI.create("udp://127.0.0.1:" + freePort()), workerSeeds, settings, spec, "netty"));
             GossipParallelMembership worker1 = GossipParallelMembership.start(new GossipParallelSettings(cluster,
                      "worker-1", URI.create("udp://127.0.0.1:" + freePort()), workerSeeds, settings, spec, "netty"));
             GossipParallelMembership worker2 = GossipParallelMembership.start(new GossipParallelSettings(cluster,
                      "worker-2", URI.create("udp://127.0.0.1:" + freePort()), workerSeeds, settings, spec, "netty"))) {

            try (GossipParallelMembership coordinator = GossipParallelMembership.startObserver(new GossipParallelSettings(
                    cluster, "coordinator", coordinatorUri, coordinatorSeeds, settings, spec, "netty"))) {
                assertCoordinatorSeesWorkersAndAssignment(coordinator, List.of("worker-0", "worker-1", "worker-2"));
                publishThreeRankEndpoints(worker0, worker1, worker2, 51000, 51001, 51002);
                eventually(() -> coordinator.findRankEndpoints("worker-2").stream()
                                .anyMatch(endpoint -> endpoint.uri().equals("http://127.0.0.1:51002")),
                        Duration.ofSeconds(10));
            }

            eventually(() -> portReleased(coordinatorPort), Duration.ofSeconds(10));

            try (GossipParallelMembership restartedCoordinator = GossipParallelMembership.startObserver(
                    new GossipParallelSettings(cluster, "coordinator", restartedCoordinatorUri, restartedCoordinatorSeeds,
                            settings, spec, "netty"))) {
                publishThreeRankEndpoints(worker0, worker1, worker2, 52000, 52001, 52002);
                assertCoordinatorSeesWorkersAndAssignment(restartedCoordinator,
                        List.of("worker-0", "worker-1", "worker-2"));
                eventually(() -> restartedCoordinator.findRankEndpoints("worker-0").stream()
                                .anyMatch(endpoint -> endpoint.uri().equals("http://127.0.0.1:52000"))
                                && restartedCoordinator.findRankEndpoints("worker-1").stream()
                                .anyMatch(endpoint -> endpoint.uri().equals("http://127.0.0.1:52001"))
                                && restartedCoordinator.findRankEndpoints("worker-2").stream()
                                .anyMatch(endpoint -> endpoint.uri().equals("http://127.0.0.1:52002")),
                        Duration.ofSeconds(10));
            }
        }
    }

    private static void assertCoordinatorSeesWorkersAndAssignment(GossipParallelMembership coordinator) throws Exception {
        assertCoordinatorSeesWorkersAndAssignment(coordinator, List.of("worker-0", "worker-1"));
    }

    private static void assertCoordinatorSeesWorkersAndAssignment(GossipParallelMembership coordinator,
            List<String> workers) throws Exception {
        eventually(() -> coordinator.candidateNodeIds().containsAll(workers),
                Duration.ofSeconds(20));
        eventually(() -> "worker-0".equals(coordinator.electedLeader()), Duration.ofSeconds(20));
        eventually(() -> coordinator.findAssignment() != null, Duration.ofSeconds(20));

        TensorParallelAssignment assignment = coordinator.findAssignment();
        assertNotNull(assignment);
        assertEquals(workers, assignment.ranks().stream().map(TensorParallelRankAssignment::nodeId).toList());
    }

    private static void publishThreeRankEndpoints(GossipParallelMembership worker0, GossipParallelMembership worker1,
            GossipParallelMembership worker2, int port0, int port1, int port2) {
        worker0.publishRankEndpoints(List.of(new TensorParallelRankEndpoint(0, "worker-0",
                "http://127.0.0.1:" + port0)));
        worker1.publishRankEndpoints(List.of(new TensorParallelRankEndpoint(1, "worker-1",
                "http://127.0.0.1:" + port1)));
        worker2.publishRankEndpoints(List.of(new TensorParallelRankEndpoint(2, "worker-2",
                "http://127.0.0.1:" + port2)));
    }

    private static GossipSettings gossipSettings() {
        GossipSettings settings = new GossipSettings();
        settings.setPersistRingState(false);
        settings.setPersistDataState(false);
        settings.setGossipInterval(100);
        settings.setCleanupInterval(1_000);
        return settings;
    }

    private static int freePort() throws IOException {
        try (ServerSocket socket = new ServerSocket(0)) {
            return socket.getLocalPort();
        }
    }

    private static boolean portReleased(int port) {
        try (ServerSocket ignored = new ServerSocket(port)) {
            return true;
        } catch (IOException e) {
            return false;
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
}
