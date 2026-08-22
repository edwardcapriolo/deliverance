package io.teknek.deliverance.model.tensorparallel;

import io.teknek.gossip.LocalMember;
import io.teknek.gossip.crdt.Crdt;
import io.teknek.gossip.crdt.OrSet;
import io.teknek.gossip.event.GossipListener;
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
    private final URI gossipUri;
    private volatile boolean closed;
    private Thread assignmentCoordinator;
    private AutoCloseable collectiveServer;
    private URI collectiveServerUri;
    private AutoModelForCausaLm.Builder rankBuilder;
    private TensorParallelWorker worker;
    private List<Integer> workerRanks = List.of();
    private int workerTensorParallelSize;
    private TensorParallelManualAssignment manualAssignmentDraft;
    private final String runtimeHost;
    private final TensorParallelTimeoutSettings timeoutSettings;
    private final TensorParallelAssignmentMode assignmentMode;
    private final boolean workerCandidate;

    private GossipParallelMembership(GossipManager gossipManager, TensorParallelDeploymentSpec deploymentSpec,
            String runtimeHost, String collectiveTransport, TensorParallelTimeoutSettings timeoutSettings,
            TensorParallelAssignmentMode assignmentMode, boolean workerCandidate) {
        this.gossipManager = gossipManager;
        this.deploymentSpec = deploymentSpec;
        this.gossipUri = gossipManager.getMyself().getUri();
        this.runtimeHost = runtimeHost;
        this.collectiveTransport = collectiveTransport;
        this.timeoutSettings = timeoutSettings;
        this.assignmentMode = assignmentMode;
        this.workerCandidate = workerCandidate;
        this.manualAssignmentDraft = new TensorParallelManualAssignment(deploymentSpec.deploymentId(), List.of());
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
                settings.uri().getHost(), settings.collectiveTransport(), settings.timeoutSettings(),
                settings.assignmentMode(), candidate);
        if (candidate) {
            membership.publishDeploymentSpec();
            membership.publishCapacity();
        }
        if (candidate || settings.assignmentMode() == TensorParallelAssignmentMode.MANUAL) {
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

    public Map<String, Object> diagnostics() {
        TensorParallelAssignment assignment = findAssignment();
        URI collectiveUri = findCollectiveUri();
        List<TensorParallelRankEndpoint> localEndpoints = findRankEndpoints(localNodeId());
        Map<String, Object> workerDiagnostics = worker == null ? Map.of(
                "started", false,
                "activeRequests", 0,
                "recentErrors", List.of(),
                "servers", List.of()) : Map.of(
                "started", true,
                "activeRequests", worker.activeRequests(),
                "recentErrors", worker.recentErrors(),
                "servers", worker.serverDiagnostics());
        return new LinkedHashMap<>(Map.ofEntries(
                Map.entry("nodeId", localNodeId()),
                Map.entry("deployment", deploymentSpec.deploymentId()),
                Map.entry("gossipUri", gossipUri.toString()),
                Map.entry("runtimeHost", runtimeHost),
                Map.entry("collectiveTransport", collectiveTransport),
                Map.entry("closed", closed),
                Map.entry("candidate", candidateNodeIds().contains(localNodeId())),
                Map.entry("liveMembers", liveMembers().stream().map(LocalMember::getId).sorted().toList()),
                Map.entry("candidates", candidateNodeIds()),
                Map.entry("leader", electedLeader() == null ? "" : electedLeader()),
                Map.entry("assignment", assignment == null ? "" : assignment.toString()),
                Map.entry("localRanks", assignment == null ? List.of() : assignment.ranksForNode(localNodeId())),
                Map.entry("servedRanks", workerRanks),
                Map.entry("collectiveUri", collectiveUri == null ? "" : collectiveUri.toString()),
                Map.entry("publishedRankEndpoints", localEndpoints),
                Map.entry("worker", workerDiagnostics)
        ));
    }

    public void registerGossipListener(GossipListener listener) {
        gossipManager.registerGossipListener(listener);
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
        publishCapacity();
    }

    public void publishCapacity() {
        PerNodeDataMessage message = new PerNodeDataMessage();
        message.setKey(deploymentSpec.capacityKey());
        message.setPayload(new TensorParallelNodeCapacity(localNodeId(), deploymentSpec.maxRanksPerNode()));
        message.setTimestamp(System.currentTimeMillis());
        message.setExpireAt(Long.MAX_VALUE);
        gossipManager.gossipPerNodeData(message);
        LOGGER.info("Published tensor-parallel capacity node={} deployment={} slots={}", localNodeId(),
                deploymentSpec.deploymentId(), deploymentSpec.maxRanksPerNode());
    }

    public TensorParallelDeploymentSpec findDeploymentSpec() {
        Object payload = findSharedData(deploymentSpec.sharedDataKey());
        return payload instanceof TensorParallelDeploymentSpec spec ? spec : null;
    }

    public List<String> candidateNodeIds() {
        return liveCapacities().stream().map(TensorParallelNodeCapacity::nodeId).sorted().toList();
    }

    public TensorParallelTopology topology() {
        List<TensorParallelNodeCapacity> capacities = liveCapacities();
        List<String> activeRankAssignments = new ArrayList<>();
        List<String> standby = new ArrayList<>();
        int availableSlots = 0;
        for (TensorParallelNodeCapacity capacity : capacities) {
            availableSlots += capacity.slots();
            if (activeRankAssignments.size() < deploymentSpec.requestedNodes()) {
                int remaining = deploymentSpec.requestedNodes() - activeRankAssignments.size();
                int ranks = Math.min(capacity.slots(), remaining);
                for (int i = 0; i < ranks; i++) {
                    activeRankAssignments.add(capacity.nodeId());
                }
            } else {
                standby.add(capacity.nodeId());
            }
        }
        return new TensorParallelTopology(deploymentSpec.deploymentId(), availableSlots,
                activeRankAssignments, standby,
                TensorParallelTopology.assignmentHash(deploymentSpec.deploymentId(), availableSlots,
                        activeRankAssignments));
    }

    private List<TensorParallelNodeCapacity> liveCapacities() {
        List<String> liveNodeIds = new ArrayList<>();
        liveNodeIds.add(localNodeId());
        for (LocalMember member : liveMembers()) {
            liveNodeIds.add(member.getId());
        }
        return liveNodeIds.stream().distinct().sorted()
                .map(this::findCapacity)
                .filter(capacity -> capacity != null)
                .toList();
    }

    private TensorParallelNodeCapacity findCapacity(String nodeId) {
        PerNodeDataMessage message = gossipManager.findPerNodeGossipData(nodeId, deploymentSpec.capacityKey());
        if (message == null || message.getPayload() == null) {
            return null;
        }
        Object payload = message.getPayload();
        if (payload instanceof TensorParallelNodeCapacity capacity) {
            return capacity;
        }
        return null;
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
            if (assignmentMode == TensorParallelAssignmentMode.MANUAL) {
                coordinateManualAssignment();
                return;
            }
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
                if (localNodeId().equals(electedLeader())) {
                    publishAssignmentAsLeader();
                }
                Thread.sleep(100);
            }
            startCollectiveServerIfLeader();
            while (!closed && findCollectiveUri() == null) {
                Thread.sleep(100);
            }
            startWorkerIfReady();
            monitorWorkerAssignment();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (RuntimeException e) {
            LOGGER.warn("Tensor-parallel assignment coordination failed", e);
        }
    }

    private void coordinateManualAssignment() throws InterruptedException {
        while (!closed && findAssignment() == null) {
            Thread.sleep(100);
        }
        if (closed) {
            return;
        }
        startCollectiveServerIfLeader();
        while (!closed && findCollectiveUri() == null) {
            Thread.sleep(100);
        }
        if (workerCandidate) {
            startWorkerIfReady();
            monitorWorkerAssignment();
        }
    }

    private synchronized void startCollectiveServerIfLeader() {
        if (closed || collectiveServer != null || !localNodeId().equals(electedLeader())) {
            return;
        }
        if (collectiveTransport.equals("netty")) {
            NettyTensorParallelCollectiveServer server = new NettyTensorParallelCollectiveServer(
                    new InetSocketAddress(runtimeHost, 0), Duration.ofSeconds(30));
            server.start();
            collectiveServer = server;
            collectiveServerUri = server.uri();
        } else {
            HttpTensorParallelCollectiveServer server = new HttpTensorParallelCollectiveServer(
                    new InetSocketAddress(runtimeHost, 0), Duration.ofSeconds(30));
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
        if (closed || rankBuilder == null) {
            return;
        }
        reconcileWorkerForCurrentAssignment();
    }

    private void monitorWorkerAssignment() throws InterruptedException {
        while (!closed) {
            Thread.sleep(500);
            reconcileWorkerIfReady();
        }
    }

    private synchronized void reconcileWorkerIfReady() {
        if (closed || rankBuilder == null) {
            return;
        }
        reconcileWorkerForCurrentAssignment();
    }

    private void reconcileWorkerForCurrentAssignment() {
        TensorParallelAssignment assignment = findAssignment();
        List<Integer> desiredRanks = assignment == null ? List.of() : assignment.ranksForNode(localNodeId());
        int desiredSize = assignment == null ? 0 : assignment.tensorParallelSize();
        if (desiredRanks.equals(workerRanks) && desiredSize == workerTensorParallelSize) {
            return;
        }
        stopWorkerAndPublishEmptyEndpoints();
        workerRanks = List.copyOf(desiredRanks);
        workerTensorParallelSize = desiredSize;
        if (desiredRanks.isEmpty()) {
            LOGGER.info("Tensor-parallel worker has no local ranks node={} deployment={}",
                    localNodeId(), deploymentSpec.deploymentId());
            return;
        }
        LOGGER.info("Starting tensor-parallel worker node={} deployment={} localRanks={} runtimeHost={}",
                localNodeId(), deploymentSpec.deploymentId(), desiredRanks, runtimeHost);
        worker = TensorParallelWorker.start(rankBuilder, this, tensorParallelCollectivesFactory(), runtimeHost);
        LOGGER.info("Started tensor-parallel worker node={} deployment={} endpoints={}",
                localNodeId(), deploymentSpec.deploymentId(), worker.endpoints());
    }

    private void stopWorkerAndPublishEmptyEndpoints() {
        if (worker != null) {
            LOGGER.info("Stopping tensor-parallel worker node={} deployment={} servedRanks={}",
                    localNodeId(), deploymentSpec.deploymentId(), workerRanks);
            worker.close();
            worker = null;
        }
        if (!workerRanks.isEmpty() || !findRankEndpoints(localNodeId()).isEmpty()) {
            publishRankEndpoints(List.of());
        }
        workerRanks = List.of();
        workerTensorParallelSize = 0;
    }

    public String electedLeader() {
        if (assignmentMode == TensorParallelAssignmentMode.MANUAL) {
            TensorParallelAssignment assignment = findAssignment();
            return assignment == null ? null : assignment.leaderNodeId();
        }
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
        if (assignmentMode == TensorParallelAssignmentMode.MANUAL) {
            throw new IllegalStateException("Automatic assignment publication is disabled in manual assignment mode");
        }
        String leader = electedLeader();
        String localNodeId = gossipManager.getMyself().getId();
        if (!localNodeId.equals(leader)) {
            throw new IllegalStateException("Only elected leader may publish assignment. local=" + localNodeId
                    + " leader=" + leader);
        }
        TensorParallelTopology topology = topology();
        if (topology.tensorParallelSize() < deploymentSpec.requestedNodes()) {
            LOGGER.info("Not publishing tensor-parallel assignment yet node={} deployment={} availableSlots={} requestedRanks={} activeNodes={}",
                    localNodeId, deploymentSpec.deploymentId(), topology.availableSlots(), deploymentSpec.requestedNodes(),
                    topology.activeNodeIds());
            return;
        }
        TensorParallelAssignment assignment = new TensorParallelAssignment(deploymentSpec.deploymentId(), leader,
                topology.tensorParallelSize(), topology.assignmentHash(), topology.rankAssignments());
        publishSharedData(deploymentSpec.assignmentKey(), assignment);
        LOGGER.info("Published tensor-parallel assignment node={} deployment={} leader={} tensorParallelSize={} ranks={}",
                localNodeId, deploymentSpec.deploymentId(), leader, assignment.tensorParallelSize(), assignment.ranks());
    }

    public synchronized TensorParallelManualAssignment assignRankManually(String nodeId, int rank) {
        if (assignmentMode != TensorParallelAssignmentMode.MANUAL) {
            throw new IllegalStateException("Manual rank assignment requires MANUAL assignment mode");
        }
        validateManualRank(nodeId, rank);
        TensorParallelManualAssignment draft = manualAssignmentDraft.withRank(nodeId, rank);
        validateManualAssignment(draft);
        manualAssignmentDraft = draft;
        publishSharedData(deploymentSpec.manualAssignmentKey(), draft);
        LOGGER.info("Published tensor-parallel manual assignment draft node={} deployment={} assignedRank={} assignedNode={} draft={}",
                localNodeId(), deploymentSpec.deploymentId(), rank, nodeId, draft.ranks());
        if (draft.complete(deploymentSpec.requestedNodes())) {
            publishManualAssignment(draft);
        }
        return draft;
    }

    public TensorParallelManualAssignment findManualAssignment() {
        if (!manualAssignmentDraft.ranks().isEmpty()) {
            return manualAssignmentDraft;
        }
        Object payload = findSharedData(deploymentSpec.manualAssignmentKey());
        if (payload instanceof TensorParallelManualAssignment assignment) {
            manualAssignmentDraft = assignment;
            return assignment;
        }
        return new TensorParallelManualAssignment(deploymentSpec.deploymentId(), List.of());
    }

    private void publishManualAssignment(TensorParallelManualAssignment draft) {
        List<String> activeRankAssignments = draft.ranks().stream()
                .map(TensorParallelRankAssignment::nodeId)
                .toList();
        int availableSlots = liveCapacities().stream().mapToInt(TensorParallelNodeCapacity::slots).sum();
        TensorParallelAssignment assignment = new TensorParallelAssignment(deploymentSpec.deploymentId(), localNodeId(),
                deploymentSpec.requestedNodes(), TensorParallelTopology.assignmentHash(deploymentSpec.deploymentId(),
                availableSlots, activeRankAssignments), draft.ranks());
        publishSharedData(deploymentSpec.assignmentKey(), assignment);
        LOGGER.info("Published tensor-parallel manual assignment node={} deployment={} tensorParallelSize={} ranks={}",
                localNodeId(), deploymentSpec.deploymentId(), assignment.tensorParallelSize(), assignment.ranks());
        startCollectiveServerIfLeader();
    }

    private void validateManualRank(String nodeId, int rank) {
        if (rank < 0 || rank >= deploymentSpec.requestedNodes()) {
            throw new IllegalArgumentException("rank must be between 0 and " + (deploymentSpec.requestedNodes() - 1));
        }
        TensorParallelNodeCapacity capacity = findLiveCapacity(nodeId);
        if (capacity == null) {
            throw new IllegalArgumentException("No live tensor-parallel capacity found for node " + nodeId);
        }
    }

    private void validateManualAssignment(TensorParallelManualAssignment draft) {
        Map<String, Integer> ranksPerNode = new LinkedHashMap<>();
        for (TensorParallelRankAssignment rank : draft.ranks()) {
            TensorParallelNodeCapacity capacity = findLiveCapacity(rank.nodeId());
            if (capacity == null) {
                throw new IllegalArgumentException("No live tensor-parallel capacity found for node " + rank.nodeId());
            }
            int count = ranksPerNode.merge(rank.nodeId(), 1, Integer::sum);
            if (count > capacity.slots()) {
                throw new IllegalArgumentException("Manual assignment gives node " + rank.nodeId() + " " + count
                        + " ranks, capacity=" + capacity.slots());
            }
        }
    }

    private TensorParallelNodeCapacity findLiveCapacity(String nodeId) {
        return liveCapacities().stream()
                .filter(capacity -> capacity.nodeId().equals(nodeId))
                .findFirst()
                .orElse(null);
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

    public TensorParallelDeploymentSpec deploymentSpec() {
        return deploymentSpec;
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
                        assignment.tensorParallelSize(), new HttpTensorParallelRankClient(URI.create(endpoint.uri()),
                                timeoutSettings.rankConnectTimeout(), timeoutSettings.rankRequestTimeout()),
                        false))
                .toList(), timeoutSettings);
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

    void mergeSharedDataForTest(String key, Crdt<?, ?> payload) {
        mergeSharedData(key, payload);
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
            stopWorkerAndPublishEmptyEndpoints();
        }
        gossipManager.shutdown();
    }
}
