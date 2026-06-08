package io.teknek.deliverance.model.tensorparallel;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

/**
 * Deterministic tensor-parallel topology derived from gossip membership data.
 */
public record TensorParallelTopology(
        String deploymentId,
        int availableSlots,
        List<String> activeRankAssignments,
        List<String> standbyNodeIds,
        String assignmentHash
) {
    public TensorParallelTopology {
        activeRankAssignments = List.copyOf(activeRankAssignments);
        standbyNodeIds = List.copyOf(standbyNodeIds);
    }

    public int rankOf(String nodeId) {
        return activeRankAssignments.indexOf(nodeId);
    }

    public List<Integer> assignedRanks(String nodeId) {
        List<Integer> ranks = new ArrayList<>();
        for (int i = 0; i < activeRankAssignments.size(); i++) {
            if (activeRankAssignments.get(i).equals(nodeId)) {
                ranks.add(i);
            }
        }
        return List.copyOf(ranks);
    }

    public List<TensorParallelRankAssignment> rankAssignments() {
        List<TensorParallelRankAssignment> assignments = new ArrayList<>();
        for (int rank = 0; rank < activeRankAssignments.size(); rank++) {
            assignments.add(new TensorParallelRankAssignment(rank, activeRankAssignments.get(rank)));
        }
        return List.copyOf(assignments);
    }

    public List<String> activeNodeIds() {
        return List.copyOf(new LinkedHashSet<>(activeRankAssignments));
    }

    public long rankCountFor(String nodeId) {
        return activeRankAssignments.stream().filter(nodeId::equals).count();
    }

    public int tensorParallelSize() {
        return activeRankAssignments.size();
    }

    public static String assignmentHash(String deploymentId, int availableSlots, List<String> activeRankAssignments) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(deploymentId.getBytes(StandardCharsets.UTF_8));
            digest.update((byte) 0);
            digest.update(Integer.toString(availableSlots).getBytes(StandardCharsets.UTF_8));
            for (String nodeId : activeRankAssignments) {
                digest.update((byte) 0);
                digest.update(nodeId.getBytes(StandardCharsets.UTF_8));
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }
}
