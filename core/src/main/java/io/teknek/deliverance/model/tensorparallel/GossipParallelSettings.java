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
        TensorParallelDeploymentSpec deploymentSpec,
        String collectiveTransport,
        String advertiseHost,
        TensorParallelTimeoutSettings timeoutSettings
) {
    public GossipParallelSettings(String cluster, String nodeId, URI uri, List<Member> seedMembers,
            GossipSettings gossipSettings, TensorParallelDeploymentSpec deploymentSpec) {
        this(cluster, nodeId, uri, seedMembers, gossipSettings, deploymentSpec, "http", null,
                TensorParallelTimeoutSettings.DEFAULT);
    }

    public GossipParallelSettings(String cluster, String nodeId, URI uri, List<Member> seedMembers,
            GossipSettings gossipSettings, TensorParallelDeploymentSpec deploymentSpec, String collectiveTransport) {
        this(cluster, nodeId, uri, seedMembers, gossipSettings, deploymentSpec, collectiveTransport, null,
                TensorParallelTimeoutSettings.DEFAULT);
    }

    public GossipParallelSettings(String cluster, String nodeId, URI uri, List<Member> seedMembers,
            GossipSettings gossipSettings, TensorParallelDeploymentSpec deploymentSpec, String collectiveTransport,
            String advertiseHost) {
        this(cluster, nodeId, uri, seedMembers, gossipSettings, deploymentSpec, collectiveTransport, advertiseHost,
                TensorParallelTimeoutSettings.DEFAULT);
    }

    public GossipParallelSettings {
        Objects.requireNonNull(cluster, "cluster");
        Objects.requireNonNull(nodeId, "nodeId");
        Objects.requireNonNull(uri, "uri");
        seedMembers = List.copyOf(Objects.requireNonNull(seedMembers, "seedMembers"));
        gossipSettings = Objects.requireNonNull(gossipSettings, "gossipSettings");
        Objects.requireNonNull(deploymentSpec, "deploymentSpec");
        collectiveTransport = Objects.requireNonNull(collectiveTransport, "collectiveTransport").toLowerCase(java.util.Locale.ROOT);
        advertiseHost = (advertiseHost == null || advertiseHost.isBlank()) ? uri.getHost() : advertiseHost;
        Objects.requireNonNull(advertiseHost, "advertiseHost");
        timeoutSettings = timeoutSettings == null ? TensorParallelTimeoutSettings.DEFAULT : timeoutSettings;
        if (!collectiveTransport.equals("http") && !collectiveTransport.equals("netty")) {
            throw new IllegalArgumentException("collectiveTransport must be http or netty");
        }
    }
}
