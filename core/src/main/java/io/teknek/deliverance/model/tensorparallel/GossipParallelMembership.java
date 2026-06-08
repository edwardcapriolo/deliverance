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

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Running gossip membership handle for one Deliverance tensor-parallel node.
 */
public class GossipParallelMembership implements AutoCloseable {
    private final GossipManager gossipManager;
    private final TensorParallelDeploymentSpec deploymentSpec;

    private GossipParallelMembership(GossipManager gossipManager, TensorParallelDeploymentSpec deploymentSpec) {
        this.gossipManager = gossipManager;
        this.deploymentSpec = deploymentSpec;
    }

    public static GossipParallelMembership start(GossipParallelSettings settings) {
        GossipManager manager = GossipManagerBuilder.newBuilder()
                .cluster(settings.cluster())
                .id(settings.nodeId())
                .uri(settings.uri())
                .gossipMembers(settings.seedMembers())
                .gossipSettings(settings.gossipSettings())
                .build();
        manager.init();
        GossipParallelMembership membership = new GossipParallelMembership(manager, settings.deploymentSpec());
        membership.publishDeploymentSpec();
        membership.publishCandidate();
        return membership;
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
    }

    public void publishCandidate() {
        mergeSharedData(deploymentSpec.candidatesKey(), new OrSet<>(gossipManager.getMyself().getId()));
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
        VoteCandidate candidate = new VoteCandidate(leaderCandidate, deploymentSpec.leaderVoteKey(), new ConcurrentHashMap<>());
        candidate.addVote(new Vote(gossipManager.getMyself().getId(), true, false, activeNodes, topology.standbyNodeIds()));
        Map<String, VoteCandidate> candidates = new LinkedHashMap<>();
        candidates.put(leaderCandidate, candidate);
        mergeSharedData(deploymentSpec.leaderVoteKey(), new MajorityVote(candidates));
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
        gossipManager.shutdown();
    }
}
