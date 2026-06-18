package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.tensorparallel.transport.HttpTensorParallelRankServer;

import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Starts local rank models assigned to one physical node and exposes them as HTTP rank endpoints.
 */
public class TensorParallelWorker implements AutoCloseable {
    private static final Logger LOGGER = LoggerFactory.getLogger(TensorParallelWorker.class);
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
                AbstractModel model = rankBuilder.buildLocalTransformerModel();
                InProcessTensorParallelRankService service = new InProcessTensorParallelRankService(model);
                HttpTensorParallelRankServer server = new HttpTensorParallelRankServer(new InetSocketAddress(bindHost, 0), service);
                server.start();
                servers.add(server);
                endpoints.add(new TensorParallelRankEndpoint(model.getTensorParallelContext().rank(), membership.localNodeId(),
                        server.uri().toString()));
                LOGGER.info("Started tensor-parallel rank server node={} rank={} size={} uri={} provider={}",
                        membership.localNodeId(), model.getTensorParallelContext().rank(),
                        model.getTensorParallelContext().size(), server.uri(), model.getTensorProviderName());
            }
            endpoints.sort(Comparator.comparingInt(TensorParallelRankEndpoint::rank));
            TensorParallelWorker worker = new TensorParallelWorker(List.copyOf(servers), List.copyOf(endpoints));
            membership.publishRankEndpoints(worker.endpoints());
            LOGGER.info("Started tensor-parallel worker node={} endpoints={}", membership.localNodeId(), worker.endpoints());
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
            LOGGER.info("Closing tensor-parallel rank server uri={}", server.uri());
            server.close();
        }
    }
}
