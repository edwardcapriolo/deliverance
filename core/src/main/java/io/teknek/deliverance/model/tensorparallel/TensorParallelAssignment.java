package io.teknek.deliverance.model.tensorparallel;

import java.util.List;
import java.util.ArrayList;
import java.util.Objects;

/**
 * Leader-published tensor-parallel rank assignment for a deployment.
 */
public record TensorParallelAssignment(
        String deploymentId,
        String leaderNodeId,
        int tensorParallelSize,
        String assignmentHash,
        List<TensorParallelRankAssignment> ranks
) {
    public TensorParallelAssignment {
        Objects.requireNonNull(deploymentId, "deploymentId");
        Objects.requireNonNull(leaderNodeId, "leaderNodeId");
        Objects.requireNonNull(assignmentHash, "assignmentHash");
        ranks = List.copyOf(Objects.requireNonNull(ranks, "ranks"));
        if (tensorParallelSize < 1) {
            throw new IllegalArgumentException("tensorParallelSize must be >= 1");
        }
        if (ranks.size() != tensorParallelSize) {
            throw new IllegalArgumentException("rank assignment count must equal tensorParallelSize");
        }
        for (int i = 0; i < ranks.size(); i++) {
            if (ranks.get(i).rank() != i) {
                throw new IllegalArgumentException("rank assignments must be contiguous and sorted by rank");
            }
        }
    }

    public List<Integer> ranksForNode(String nodeId) {
        List<Integer> result = new ArrayList<>();
        for (TensorParallelRankAssignment assignment : ranks) {
            if (assignment.nodeId().equals(nodeId)) {
                result.add(assignment.rank());
            }
        }
        return List.copyOf(result);
    }

    public boolean assignsEveryRank() {
        if (ranks.size() != tensorParallelSize) {
            return false;
        }
        for (int i = 0; i < ranks.size(); i++) {
            if (ranks.get(i).rank() != i) {
                return false;
            }
        }
        return true;
    }

    public boolean matchesTopology(TensorParallelTopology topology) {
        return deploymentId.equals(topology.deploymentId())
                && tensorParallelSize == topology.tensorParallelSize()
                && assignmentHash.equals(topology.assignmentHash())
                && ranks.equals(topology.rankAssignments());
    }
}
