package io.teknek.reckoning.model.tensorparallel;

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

    public int minimumPhysicalNodes() {
        return (requestedNodes + maxRanksPerNode - 1) / maxRanksPerNode;
    }

    public String sharedDataKey() {
        return "reckoning.tp.deployment." + deploymentId;
    }

    public String candidatesKey() {
        return "reckoning.tp.candidates." + deploymentId;
    }

    public String capacityKey() {
        return "reckoning.tp.capacity." + deploymentId;
    }

    public String leaderVoteKey() {
        return "reckoning.tp.leaderVote." + deploymentId;
    }

    public String assignmentKey() {
        return "reckoning.tp.assignment." + deploymentId;
    }

    public String manualAssignmentKey() {
        return "reckoning.tp.manualAssignment." + deploymentId;
    }

    public String rankEndpointsKey() {
        return "reckoning.tp.rankEndpoints." + deploymentId;
    }

    public String collectiveUriKey() {
        return "reckoning.tp.collectiveUri." + deploymentId;
    }
}
