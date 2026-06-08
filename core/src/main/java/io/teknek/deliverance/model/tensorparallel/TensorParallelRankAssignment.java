package io.teknek.deliverance.model.tensorparallel;

import java.util.Objects;

/**
 * Assignment of one tensor-parallel rank to one physical node.
 */
public record TensorParallelRankAssignment(int rank, String nodeId) {
    public TensorParallelRankAssignment {
        if (rank < 0) {
            throw new IllegalArgumentException("rank must be >= 0");
        }
        Objects.requireNonNull(nodeId, "nodeId");
    }
}
