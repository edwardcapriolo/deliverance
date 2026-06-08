package io.teknek.deliverance.model.tensorparallel;

import java.util.Objects;

/**
 * HTTP endpoint metadata for one tensor-parallel rank served by a physical node.
 */
public record TensorParallelRankEndpoint(int rank, String nodeId, String uri) {
    public TensorParallelRankEndpoint {
        if (rank < 0) {
            throw new IllegalArgumentException("rank must be >= 0");
        }
        Objects.requireNonNull(nodeId, "nodeId");
        Objects.requireNonNull(uri, "uri");
    }
}
