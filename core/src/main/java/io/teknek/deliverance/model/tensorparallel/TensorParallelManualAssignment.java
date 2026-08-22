package io.teknek.deliverance.model.tensorparallel;

import java.util.Comparator;
import java.util.List;
import java.util.Objects;

public record TensorParallelManualAssignment(String deploymentId, List<TensorParallelRankAssignment> ranks) {
    public TensorParallelManualAssignment {
        Objects.requireNonNull(deploymentId, "deploymentId");
        ranks = ranks == null ? List.of() : ranks.stream()
                .sorted(Comparator.comparingInt(TensorParallelRankAssignment::rank))
                .toList();
    }

    public TensorParallelManualAssignment withRank(String nodeId, int rank) {
        List<TensorParallelRankAssignment> updated = ranks.stream()
                .filter(existing -> existing.rank() != rank)
                .collect(java.util.stream.Collectors.toCollection(java.util.ArrayList::new));
        updated.add(new TensorParallelRankAssignment(rank, nodeId));
        return new TensorParallelManualAssignment(deploymentId, updated);
    }

    public TensorParallelManualAssignment withoutRank(int rank) {
        return new TensorParallelManualAssignment(deploymentId, ranks.stream()
                .filter(existing -> existing.rank() != rank)
                .toList());
    }

    public boolean complete(int tensorParallelSize) {
        if (ranks.size() != tensorParallelSize) {
            return false;
        }
        for (int rank = 0; rank < tensorParallelSize; rank++) {
            if (ranks.get(rank).rank() != rank) {
                return false;
            }
        }
        return true;
    }
}
