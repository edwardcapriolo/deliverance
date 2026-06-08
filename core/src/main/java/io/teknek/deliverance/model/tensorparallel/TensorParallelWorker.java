package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.tensorparallel.transport.HttpTensorParallelRankServer;

import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;

/**
 * Starts local rank models assigned to one physical node and exposes them as HTTP rank endpoints.
 */
public class TensorParallelWorker implements AutoCloseable {
    private final List<HttpTensorParallelRankServer> servers;
    private final List<TensorParallelRankEndpoint> endpoints;

    private TensorParallelWorker(List<HttpTensorParallelRankServer> servers,
            List<TensorParallelRankEndpoint> endpoints) {
        this.servers = servers;
        this.endpoints = endpoints;
    }

    public static TensorParallelWorker start(AutoModelForCausaLm.Builder builder,
            GossipParallelMembership membership,
            Function<TensorParallelContext, TensorParallelCollectives> collectivesFactory,
            String bindHost) {
        List<AutoModelForCausaLm.Builder> rankBuilders = builder.localAssignedRankBuilders(membership, collectivesFactory);
        List<HttpTensorParallelRankServer> servers = new ArrayList<>();
        List<TensorParallelRankEndpoint> endpoints = new ArrayList<>();
        try {
            for (AutoModelForCausaLm.Builder rankBuilder : rankBuilders) {
                AbstractModel model = rankBuilder.build();
                InProcessTensorParallelRankService service = new InProcessTensorParallelRankService(model);
                HttpTensorParallelRankServer server = new HttpTensorParallelRankServer(new InetSocketAddress(bindHost, 0), service);
                server.start();
                servers.add(server);
                endpoints.add(new TensorParallelRankEndpoint(model.getTensorParallelContext().rank(), membership.localNodeId(),
                        server.uri().toString()));
            }
            endpoints.sort(Comparator.comparingInt(TensorParallelRankEndpoint::rank));
            TensorParallelWorker worker = new TensorParallelWorker(List.copyOf(servers), List.copyOf(endpoints));
            membership.publishRankEndpoints(worker.endpoints());
            return worker;
        } catch (RuntimeException e) {
            for (HttpTensorParallelRankServer server : servers) {
                server.close();
            }
            throw e;
        }
    }

    public List<TensorParallelRankEndpoint> endpoints() {
        return endpoints;
    }

    @Override
    public void close() {
        for (HttpTensorParallelRankServer server : servers) {
            server.close();
        }
    }
}
