package io.teknek.deliverance.model.tensorparallel;

import java.util.Objects;

/**
 * Desired tensor-parallel deployment metadata shared through cluster membership.
 */
public record TensorParallelDeploymentSpec(String deploymentId, int requestedNodes, int maxRanksPerNode) {
    public TensorParallelDeploymentSpec {
        Objects.requireNonNull(deploymentId, "deploymentId");
        if (requestedNodes < 1) {
            throw new IllegalArgumentException("requestedNodes must be >= 1");
        }
        if (maxRanksPerNode < 1) {
            throw new IllegalArgumentException("maxRanksPerNode must be >= 1");
        }
    }

    public TensorParallelDeploymentSpec(String deploymentId, int requestedNodes) {
        this(deploymentId, requestedNodes, 1);
    }

    public String sharedDataKey() {
        return "deliverance.tp.deployment." + deploymentId;
    }

    public String candidatesKey() {
        return "deliverance.tp.candidates." + deploymentId;
    }

    public String leaderVoteKey() {
        return "deliverance.tp.leaderVote." + deploymentId;
    }

    public String assignmentKey() {
        return "deliverance.tp.assignment." + deploymentId;
    }

    public String rankEndpointsKey() {
        return "deliverance.tp.rankEndpoints." + deploymentId;
    }
}
