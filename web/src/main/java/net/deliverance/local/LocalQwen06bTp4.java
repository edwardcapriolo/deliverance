package net.deliverance.local;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.tensorparallel.GossipParallelSettings;
import io.teknek.deliverance.model.tensorparallel.TensorParallelDeploymentSpec;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.gossip.GossipSettings;
import io.teknek.gossip.Member;
import io.teknek.gossip.RemoteMember;

import java.net.URI;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;

final class LocalQwen06bTp4 {
    static final String CLUSTER = "deliverance-qwen06b-local-tp4";
    static final String DEPLOYMENT = "qwen06b-local";
    static final String OWNER = "edwardcapriolo";
    static final String MODEL = "Qwen3-0.6B-JQ4";
    static final String HOST = "127.0.0.1";
    static final int WORKER_0_PORT = 42604;
    static final int WORKER_1_PORT = 42605;
    static final int COORDINATOR_PORT = 42606;
    static final int WEB_PORT = 18087;
    static final int TP_SIZE = 4;
    static final int MAX_RANKS_PER_WORKER = 2;
    static final int POOL_SIZE = 1;
    static final String COLLECTIVE_TRANSPORT = "netty";

    private LocalQwen06bTp4() {
    }

    static String workerUri(int port) {
        return "udp://" + HOST + ":" + port;
    }

    static String seed(String nodeId, int port) {
        return nodeId + "=" + workerUri(port);
    }

    static List<Member> seedMembers() {
        return List.of(
                new RemoteMember(CLUSTER, URI.create(workerUri(WORKER_0_PORT)), "worker-0"),
                new RemoteMember(CLUSTER, URI.create(workerUri(WORKER_1_PORT)), "worker-1"));
    }

    static GossipSettings gossipSettings() {
        GossipSettings settings = new GossipSettings();
        settings.setPersistRingState(false);
        settings.setPersistDataState(false);
        settings.setGossipInterval(100);
        settings.setCleanupInterval(2_000);
        return settings;
    }

    static TensorParallelDeploymentSpec deploymentSpec() {
        return new TensorParallelDeploymentSpec(DEPLOYMENT, TP_SIZE, MAX_RANKS_PER_WORKER);
    }

    static void runWorker(String nodeId, int port) throws InterruptedException {
        MetricRegistry metrics = new MetricRegistry();
        TensorAllocator allocator = new ArrayQueueTensorAllocator(metrics);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(POOL_SIZE,
                ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, true));
        ConfigurableTensorProvider provider = new ConfigurableTensorProvider(
                new PanamaTensorOperations(io.teknek.deliverance.tensor.operations.MachineSpec.VECTOR_TYPE, allocator, pool));
        GossipParallelSettings settings = new GossipParallelSettings(CLUSTER, nodeId, URI.create(workerUri(port)),
                seedMembers(), gossipSettings(), deploymentSpec(), COLLECTIVE_TRANSPORT);
        AbstractModel model = AutoModelForCausaLm.newBuilder(new ModelFetcher(OWNER, MODEL))
                .withMetricRegistry(metrics)
                .withTensorAllocator(allocator)
                .withTensorProvider(provider)
                .withWrappedForkJoinPool(pool)
                .withWorkingMemoryType(DType.F32)
                .withWorkingQuantType(DType.I8)
                .withOutputHeadQuantization(DType.Q4)
                .withParallelSettings(settings)
                .buildAbstractModel();
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            model.close();
            pool.close();
        }, "qwen06b-tp4-worker-shutdown-" + nodeId));
        new CountDownLatch(1).await();
    }

    static Map<String, Object> coordinatorProperties() {
        Map<String, Object> properties = new LinkedHashMap<>();
        properties.put("server.port", WEB_PORT);
        properties.put("deliverance.tensor.operations.type", "simd");
        properties.put("deliverance-model.configs[0].model-owner", OWNER);
        properties.put("deliverance-model.configs[0].model-name", MODEL);
        properties.put("deliverance-model.configs[0].inference-type", "GENERATION");
        properties.put("deliverance-model.configs[0].output-head-quantization", "Q4");
        properties.put("deliverance-model.configs[0].tensor-parallel.enabled", true);
        properties.put("deliverance-model.configs[0].tensor-parallel.cluster", CLUSTER);
        properties.put("deliverance-model.configs[0].tensor-parallel.node-id", "coordinator");
        properties.put("deliverance-model.configs[0].tensor-parallel.uri", workerUri(COORDINATOR_PORT));
        properties.put("deliverance-model.configs[0].tensor-parallel.seeds[0]", seed("worker-0", WORKER_0_PORT));
        properties.put("deliverance-model.configs[0].tensor-parallel.seeds[1]", seed("worker-1", WORKER_1_PORT));
        properties.put("deliverance-model.configs[0].tensor-parallel.deployment", DEPLOYMENT);
        properties.put("deliverance-model.configs[0].tensor-parallel.collective-transport", COLLECTIVE_TRANSPORT);
        properties.put("deliverance-model.configs[0].tensor-parallel.size", TP_SIZE);
        properties.put("deliverance-model.configs[0].tensor-parallel.max-ranks-per-worker", MAX_RANKS_PER_WORKER);
        properties.put("deliverance-model.configs[0].tensor-parallel.ready-timeout-seconds", 180);
        properties.put("deliverance-model.configs[0].tensor-parallel.rank-endpoint-timeout-seconds", 300);
        properties.put("deliverance.kv.prefix.max-entries", 0);
        properties.put("deliverance.kv.prefix.max-tokens", 0);
        return properties;
    }
}
