package io.teknek.deliverance.model.tensorparallel;

import io.teknek.gossip.GossipSettings;
import io.teknek.gossip.Member;

import java.net.URI;
import java.util.List;
import java.util.Objects;

/**
 * Minimal gossip configuration for a Deliverance tensor-parallel membership node.
 */
public record GossipParallelSettings(
        String cluster,
        String nodeId,
        URI uri,
        List<Member> seedMembers,
        GossipSettings gossipSettings,
        TensorParallelDeploymentSpec deploymentSpec
) {
    public GossipParallelSettings {
        Objects.requireNonNull(cluster, "cluster");
        Objects.requireNonNull(nodeId, "nodeId");
        Objects.requireNonNull(uri, "uri");
        seedMembers = List.copyOf(Objects.requireNonNull(seedMembers, "seedMembers"));
        gossipSettings = Objects.requireNonNull(gossipSettings, "gossipSettings");
        Objects.requireNonNull(deploymentSpec, "deploymentSpec");
    }
}
