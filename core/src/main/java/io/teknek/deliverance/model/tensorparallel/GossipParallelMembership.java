package io.teknek.deliverance.model.tensorparallel;

import io.teknek.gossip.LocalMember;
import io.teknek.gossip.crdt.Crdt;
import io.teknek.gossip.crdt.OrSet;
import io.teknek.gossip.lock.vote.MajorityVote;
import io.teknek.gossip.lock.vote.Vote;
import io.teknek.gossip.lock.vote.VoteCandidate;
import io.teknek.gossip.manager.GossipManager;
import io.teknek.gossip.manager.GossipManagerBuilder;
import io.teknek.gossip.model.SharedDataMessage;
import io.teknek.gossip.model.PerNodeDataMessage;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.tensorparallel.transport.HttpTensorParallelCollectiveServer;
import io.teknek.deliverance.model.tensorparallel.transport.HttpTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.transport.HttpTensorParallelRankClient;
import io.teknek.deliverance.model.tensorparallel.transport.NettyTensorParallelCollectiveServer;
import io.teknek.deliverance.model.tensorparallel.transport.NettyTensorParallelCollectives;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.InetSocketAddress;
import java.net.URI;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * Running gossip membership handle for one Deliverance tensor-parallel node.
 */
public class GossipParallelMembership implements AutoCloseable {
    private static final Logger LOGGER = LoggerFactory.getLogger(GossipParallelMembership.class);
    private final GossipManager gossipManager;
    private final TensorParallelDeploymentSpec deploymentSpec;
    private final String collectiveTransport;
    private volatile boolean closed;
    private Thread assignmentCoordinator;
    private AutoCloseable collectiveServer;
    private URI collectiveServerUri;
    private AutoModelForCausaLm.Builder rankBuilder;
    private TensorParallelWorker worker;
    private final String rankBindHost;

    private GossipParallelMembership(GossipManager gossipManager, TensorParallelDeploymentSpec deploymentSpec,
            String rankBindHost, String collectiveTransport) {
        this.gossipManager = gossipManager;
        this.deploymentSpec = deploymentSpec;
        this.rankBindHost = rankBindHost;
        this.collectiveTransport = collectiveTransport;
    }

    public static GossipParallelMembership start(GossipParallelSettings settings) {
        return start(settings, true);
    }

    /**
     * Joins the gossip cluster as a read-only observer for coordinator/debug tooling.
     *
     * <p>Observers can discover assignments, collectives, and rank endpoints, but they do not publish themselves as rank
     * candidates and therefore do not affect leader election or tensor-parallel rank placement.</p>
     */
    public static GossipParallelMembership startObserver(GossipParallelSettings settings) {
        return start(settings, false);
    }

    private static GossipParallelMembership start(GossipParallelSettings settings, boolean candidate) {
        LOGGER.info("Starting tensor-parallel gossip membership cluster={} node={} uri={} deployment={} requestedRanks={} maxRanksPerNode={}",
                settings.cluster(), settings.nodeId(), settings.uri(), settings.deploymentSpec().deploymentId(),
                settings.deploymentSpec().requestedNodes(), settings.deploymentSpec().maxRanksPerNode());
        GossipManager manager = GossipManagerBuilder.newBuilder()
                .cluster(settings.cluster())
                .id(settings.nodeId())
                .uri(settings.uri())
                .gossipMembers(settings.seedMembers())
                .gossipSettings(settings.gossipSettings())
                .build();
        manager.init();
        GossipParallelMembership membership = new GossipParallelMembership(manager, settings.deploymentSpec(),
                settings.uri().getHost(), settings.collectiveTransport());
        if (candidate) {
            membership.publishDeploymentSpec();
            membership.publishCandidate();
            membership.startAssignmentCoordinator();
        }
        LOGGER.info("Started tensor-parallel gossip membership cluster={} node={} uri={}",
                settings.cluster(), settings.nodeId(), settings.uri());
        return membership;
    }

    public synchronized void startWorkerWhenReady(AutoModelForCausaLm.Builder rankBuilder) {
        this.rankBuilder = rankBuilder;
        LOGGER.info("Tensor-parallel worker requested node={} deployment={}", localNodeId(), deploymentSpec.deploymentId());
        notifyAll();
    }

    public List<LocalMember> liveMembers() {
        return gossipManager.getLiveMembers();
    }

    public void publishSharedData(String key, Object payload) {
        SharedDataMessage message = new SharedDataMessage();
        message.setKey(key);
        message.setPayload(payload);
        message.setTimestamp(System.currentTimeMillis());
        message.setExpireAt(Long.MAX_VALUE);
        gossipManager.gossipSharedData(message);
    }

    public void publishDeploymentSpec() {
        publishSharedData(deploymentSpec.sharedDataKey(), deploymentSpec);
        LOGGER.info("Published tensor-parallel deployment spec node={} deployment={} requestedRanks={} maxRanksPerNode={}",
                localNodeId(), deploymentSpec.deploymentId(), deploymentSpec.requestedNodes(), deploymentSpec.maxRanksPerNode());
    }

    public void publishCandidate() {
        mergeSharedData(deploymentSpec.candidatesKey(), new OrSet<>(gossipManager.getMyself().getId()));
        LOGGER.info("Published tensor-parallel candidate node={} deployment={}", localNodeId(), deploymentSpec.deploymentId());
    }

    public TensorParallelDeploymentSpec findDeploymentSpec() {
        Object payload = findSharedData(deploymentSpec.sharedDataKey());
        return payload instanceof TensorParallelDeploymentSpec spec ? spec : null;
    }

    public List<String> candidateNodeIds() {
        Crdt<?, ?> crdt = gossipManager.findCrdt(deploymentSpec.candidatesKey());
        if (!(crdt instanceof OrSet<?> set)) {
            return List.of();
        }
        List<String> nodes = new ArrayList<>();
        for (Object value : set.value()) {
            nodes.add(String.valueOf(value));
        }
        Collections.sort(nodes);
        return nodes;
    }

    public TensorParallelTopology topology() {
        List<String> candidates = candidateNodeIds();
        List<String> activeRankAssignments = new ArrayList<>();
        List<String> standby = new ArrayList<>();
        for (String candidate : candidates) {
            if (activeRankAssignments.size() < deploymentSpec.requestedNodes()) {
                int remaining = deploymentSpec.requestedNodes() - activeRankAssignments.size();
                int ranks = Math.min(deploymentSpec.maxRanksPerNode(), remaining);
                for (int i = 0; i < ranks; i++) {
                    activeRankAssignments.add(candidate);
                }
            } else {
                standby.add(candidate);
            }
        }
        return new TensorParallelTopology(deploymentSpec.deploymentId(), deploymentSpec.requestedNodes(),
                activeRankAssignments, standby,
                TensorParallelTopology.assignmentHash(deploymentSpec.deploymentId(), deploymentSpec.requestedNodes(),
                        activeRankAssignments));
    }

    public void voteForLeader() {
        TensorParallelTopology topology = topology();
        List<String> activeNodes = topology.activeNodeIds();
        if (!activeNodes.contains(gossipManager.getMyself().getId())) {
            return;
        }
        String leaderCandidate = activeNodes.get(0);
        LOGGER.info("Voting for tensor-parallel leader node={} deployment={} leaderCandidate={} activeNodes={} standbyNodes={}",
                localNodeId(), deploymentSpec.deploymentId(), leaderCandidate, activeNodes, topology.standbyNodeIds());
        VoteCandidate candidate = new VoteCandidate(leaderCandidate, deploymentSpec.leaderVoteKey(), new ConcurrentHashMap<>());
        candidate.addVote(new Vote(gossipManager.getMyself().getId(), true, false, activeNodes, topology.standbyNodeIds()));
        Map<String, VoteCandidate> candidates = new LinkedHashMap<>();
        candidates.put(leaderCandidate, candidate);
        mergeSharedData(deploymentSpec.leaderVoteKey(), new MajorityVote(candidates));
    }

    /**
     * Publishes this node's leader vote only if a leader has not already been elected.
     *
     * <p>Callers should wait until the expected candidates are visible before invoking this method. Starting an election
     * too early can vote against an incomplete topology. This method only avoids redundant votes once an election has
     * already converged.</p>
     */
    public void voteForLeaderIfNeeded() {
        if (electedLeader() == null) {
            voteForLeader();
        }
    }

    private void startAssignmentCoordinator() {
        assignmentCoordinator = new Thread(this::coordinateAssignment,
                "deliverance-tp-assignment-" + deploymentSpec.deploymentId() + "-" + localNodeId());
        assignmentCoordinator.setDaemon(true);
        assignmentCoordinator.start();
    }

    private void coordinateAssignment() {
        try {
            while (!closed && candidateNodeIds().size() < deploymentSpec.minimumPhysicalNodes()) {
                Thread.sleep(100);
            }
            if (closed) {
                return;
            }
            voteForLeaderIfNeeded();
            while (!closed && electedLeader() == null) {
                Thread.sleep(100);
                voteForLeaderIfNeeded();
            }
            if (localNodeId().equals(electedLeader())) {
                publishAssignmentAsLeader();
            }
            while (!closed && findAssignment() == null) {
                Thread.sleep(100);
            }
            startCollectiveServerIfLeader();
            while (!closed && findCollectiveUri() == null) {
                Thread.sleep(100);
            }
            startWorkerIfReady();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (RuntimeException e) {
            LOGGER.warn("Tensor-parallel assignment coordination failed", e);
        }
    }

    private synchronized void startCollectiveServerIfLeader() {
        if (closed || collectiveServer != null || !localNodeId().equals(electedLeader())) {
            return;
        }
        if (collectiveTransport.equals("netty")) {
            NettyTensorParallelCollectiveServer server = new NettyTensorParallelCollectiveServer(
                    new InetSocketAddress(rankBindHost, 0), Duration.ofSeconds(30));
            server.start();
            collectiveServer = server;
            collectiveServerUri = server.uri();
        } else {
            HttpTensorParallelCollectiveServer server = new HttpTensorParallelCollectiveServer(
                    new InetSocketAddress(rankBindHost, 0), Duration.ofSeconds(30));
            server.start();
            collectiveServer = server;
            collectiveServerUri = server.uri();
        }
        publishSharedData(deploymentSpec.collectiveUriKey(), collectiveServerUri.toString());
        LOGGER.info("Started tensor-parallel collective server node={} deployment={} uri={}",
                localNodeId(), deploymentSpec.deploymentId(), collectiveServerUri);
    }

    private synchronized void startWorkerIfReady() throws InterruptedException {
        while (!closed && rankBuilder == null) {
            wait(100);
        }
        if (closed || worker != null || rankBuilder == null) {
            return;
        }
        LOGGER.info("Starting tensor-parallel worker node={} deployment={} localRanks={} bindHost={}",
                localNodeId(), deploymentSpec.deploymentId(), localRanks(), rankBindHost);
        worker = TensorParallelWorker.start(rankBuilder, this, tensorParallelCollectivesFactory(), rankBindHost);
        LOGGER.info("Started tensor-parallel worker node={} deployment={} endpoints={}",
                localNodeId(), deploymentSpec.deploymentId(), worker.endpoints());
    }

    public String electedLeader() {
        Crdt<?, ?> crdt = gossipManager.findCrdt(deploymentSpec.leaderVoteKey());
        if (!(crdt instanceof MajorityVote vote)) {
            return null;
        }
        List<String> activeNodes = topology().activeNodeIds();
        for (VoteCandidate candidate : vote.value().values()) {
            if (hasVotesFromAllActiveNodes(candidate, activeNodes)) {
                return candidate.getCandidateNodeId();
            }
        }
        return null;
    }

    public void publishAssignmentAsLeader() {
        String leader = electedLeader();
        String localNodeId = gossipManager.getMyself().getId();
        if (!localNodeId.equals(leader)) {
            throw new IllegalStateException("Only elected leader may publish assignment. local=" + localNodeId
                    + " leader=" + leader);
        }
        TensorParallelTopology topology = topology();
        TensorParallelAssignment assignment = new TensorParallelAssignment(deploymentSpec.deploymentId(), leader,
                topology.tensorParallelSize(), topology.assignmentHash(), topology.rankAssignments());
        publishSharedData(deploymentSpec.assignmentKey(), assignment);
        LOGGER.info("Published tensor-parallel assignment node={} deployment={} leader={} tensorParallelSize={} ranks={}",
                localNodeId, deploymentSpec.deploymentId(), leader, assignment.tensorParallelSize(), assignment.ranks());
    }

    public TensorParallelAssignment findAssignment() {
        Object payload = findSharedData(deploymentSpec.assignmentKey());
        return payload instanceof TensorParallelAssignment assignment ? assignment : null;
    }

    public TensorParallelAssignment requireAssignment() {
        TensorParallelAssignment assignment = findAssignment();
        if (assignment == null) {
            throw new IllegalStateException("No tensor-parallel assignment found for deployment "
                    + deploymentSpec.deploymentId());
        }
        return assignment;
    }

    public URI findCollectiveUri() {
        Object payload = findSharedData(deploymentSpec.collectiveUriKey());
        return payload == null ? null : URI.create(String.valueOf(payload));
    }

    public URI requireCollectiveUri() {
        URI uri = findCollectiveUri();
        if (uri == null) {
            throw new IllegalStateException("No tensor-parallel collective URI found for deployment "
                    + deploymentSpec.deploymentId());
        }
        return uri;
    }

    public Function<TensorParallelContext, TensorParallelCollectives> tensorParallelCollectivesFactory() {
        URI uri = requireCollectiveUri();
        if ("netty".equalsIgnoreCase(uri.getScheme())) {
            return context -> new NettyTensorParallelCollectives(context, uri);
        }
        return context -> new HttpTensorParallelCollectives(context, uri);
    }

    public TensorParallelGenerationGroup openGenerationGroup() {
        TensorParallelAssignment assignment = requireAssignment();
        List<TensorParallelRankEndpoint> endpoints = rankEndpointsForAssignment();
        LOGGER.info("Opening tensor-parallel generation group node={} deployment={} tensorParallelSize={} endpoints={}",
                localNodeId(), deploymentSpec.deploymentId(), assignment.tensorParallelSize(), endpoints);
        return TensorParallelGenerationGroup.fromEndpoints(endpoints.stream()
                .map(endpoint -> new TensorParallelGenerationGroup.RankEndpoint(endpoint.rank(),
                        assignment.tensorParallelSize(), new HttpTensorParallelRankClient(URI.create(endpoint.uri())),
                        false))
                .toList());
    }

    public List<Integer> localRanks() {
        return requireAssignment().ranksForNode(gossipManager.getMyself().getId());
    }

    public String localNodeId() {
        return gossipManager.getMyself().getId();
    }

    public void publishRankEndpoints(List<TensorParallelRankEndpoint> endpoints) {
        PerNodeDataMessage message = new PerNodeDataMessage();
        message.setKey(deploymentSpec.rankEndpointsKey());
        message.setPayload(List.copyOf(endpoints));
        message.setTimestamp(System.currentTimeMillis());
        message.setExpireAt(Long.MAX_VALUE);
        gossipManager.gossipPerNodeData(message);
        LOGGER.info("Published tensor-parallel rank endpoints node={} deployment={} endpoints={}",
                localNodeId(), deploymentSpec.deploymentId(), endpoints);
    }

    @SuppressWarnings("unchecked")
    public List<TensorParallelRankEndpoint> findRankEndpoints(String nodeId) {
        PerNodeDataMessage message = gossipManager.findPerNodeGossipData(nodeId, deploymentSpec.rankEndpointsKey());
        if (message == null || message.getPayload() == null) {
            return List.of();
        }
        return (List<TensorParallelRankEndpoint>) message.getPayload();
    }

    public List<TensorParallelRankEndpoint> rankEndpointsForAssignment() {
        TensorParallelAssignment assignment = requireAssignment();
        List<TensorParallelRankEndpoint> endpoints = new ArrayList<>();
        for (String nodeId : assignment.ranks().stream().map(TensorParallelRankAssignment::nodeId).distinct().toList()) {
            endpoints.addAll(findRankEndpoints(nodeId));
        }
        endpoints.sort(java.util.Comparator.comparingInt(TensorParallelRankEndpoint::rank));
        if (endpoints.size() != assignment.tensorParallelSize()) {
            throw new IllegalStateException("Expected " + assignment.tensorParallelSize() + " rank endpoints but found "
                    + endpoints.size());
        }
        for (int i = 0; i < endpoints.size(); i++) {
            TensorParallelRankEndpoint endpoint = endpoints.get(i);
            TensorParallelRankAssignment expected = assignment.ranks().get(i);
            if (endpoint.rank() != expected.rank() || !endpoint.nodeId().equals(expected.nodeId())) {
                throw new IllegalStateException("Rank endpoint does not match assignment at rank " + i);
            }
        }
        return List.copyOf(endpoints);
    }

    public boolean assignmentMatchesLocalTopology() {
        TensorParallelAssignment assignment = findAssignment();
        return assignment != null && assignment.matchesTopology(topology());
    }

    private static boolean hasVotesFromAllActiveNodes(VoteCandidate candidate, List<String> activeNodes) {
        for (String nodeId : activeNodes) {
            Vote vote = candidate.getVotes().get(nodeId);
            if (vote == null || !Boolean.TRUE.equals(vote.getVoteValue())) {
                return false;
            }
        }
        return true;
    }

    public Object findSharedData(String key) {
        SharedDataMessage message = gossipManager.findSharedGossipData(key);
        return message == null ? null : message.getPayload();
    }

    private void mergeSharedData(String key, Crdt<?, ?> payload) {
        SharedDataMessage message = new SharedDataMessage();
        message.setKey(key);
        message.setPayload(payload);
        message.setTimestamp(System.currentTimeMillis());
        message.setExpireAt(Long.MAX_VALUE);
        gossipManager.merge(message);
    }

    @Override
    public void close() {
        closed = true;
        if (assignmentCoordinator != null) {
            assignmentCoordinator.interrupt();
        }
        if (collectiveServer != null) {
            try {
                collectiveServer.close();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            collectiveServer = null;
        }
        if (worker != null) {
            worker.close();
            worker = null;
        }
        gossipManager.shutdown();
    }
}
