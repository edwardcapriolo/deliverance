package net.deliverance.local;

import io.teknek.deliverance.benchmark.TpLocalCluster;

import java.util.ArrayList;
import java.util.List;

final class LocalQwen06bTp4 {
    static final String OWNER = "edwardcapriolo";
    static final String MODEL = "Qwen3-0.6B-JQ4";
    static final String CLUSTER = "qwen06b-local";
    static final String DEPLOYMENT = "qwen06b-local";
    static final String WORKER_0 = "worker-0";
    static final String WORKER_1 = "worker-1";
    static final String WORKER_0_URI = "udp://127.0.0.1:42601";
    static final String WORKER_1_URI = "udp://127.0.0.1:42602";
    static final String COORDINATOR_URI = "udp://127.0.0.1:42606";
    static final int WEB_PORT = 18087;

    private LocalQwen06bTp4() {
    }

    static void runWorker(String nodeId, String uri) throws Exception {
        TpLocalCluster.main(new String[]{
                "--role", "worker",
                "--cluster", CLUSTER,
                "--node-id", nodeId,
                "--uri", uri,
                "--seed", WORKER_0 + "=" + WORKER_0_URI,
                "--seed", WORKER_1 + "=" + WORKER_1_URI,
                "--deployment", DEPLOYMENT,
                "--collective-transport", "netty",
                "--tensor-parallel-size", "4",
                "--max-ranks-per-worker", "2",
                "--owner", OWNER,
                "--model", MODEL,
                "--pool-size", "1",
                "--tensor-operations", "jvector",
                "--working-dtype", "F32",
                "--working-qtype", "I8",
                "--output-head-quantization", "Q4",
                "--no-profile-stages"
        });
    }

    static String[] webArgs(String[] args) {
        List<String> springArgs = new ArrayList<>();
        springArgs.add("--server.port=" + WEB_PORT);
        springArgs.add("--deliverance.tensor.operations.type=jvector");
        springArgs.add("--deliverance-model.configs[0].model-owner=" + OWNER);
        springArgs.add("--deliverance-model.configs[0].model-name=" + MODEL);
        springArgs.add("--deliverance-model.configs[0].inference-type=GENERATION");
        springArgs.add("--deliverance-model.configs[0].output-head-quantization=Q4");
        springArgs.add("--deliverance-model.configs[0].tensor-parallel.enabled=true");
        springArgs.add("--deliverance-model.configs[0].tensor-parallel.cluster=" + CLUSTER);
        springArgs.add("--deliverance-model.configs[0].tensor-parallel.node-id=coordinator");
        springArgs.add("--deliverance-model.configs[0].tensor-parallel.uri=" + COORDINATOR_URI);
        springArgs.add("--deliverance-model.configs[0].tensor-parallel.seeds[0]=" + WORKER_0 + "=" + WORKER_0_URI);
        springArgs.add("--deliverance-model.configs[0].tensor-parallel.seeds[1]=" + WORKER_1 + "=" + WORKER_1_URI);
        springArgs.add("--deliverance-model.configs[0].tensor-parallel.deployment=" + DEPLOYMENT);
        springArgs.add("--deliverance-model.configs[0].tensor-parallel.collective-transport=netty");
        springArgs.add("--deliverance-model.configs[0].tensor-parallel.size=4");
        springArgs.add("--deliverance-model.configs[0].tensor-parallel.max-ranks-per-worker=2");
        springArgs.add("--deliverance-model.configs[0].tensor-parallel.output-head-quantization=Q4");
        springArgs.add("--deliverance-model.configs[0].tensor-parallel.ready-timeout-seconds=120");
        springArgs.add("--deliverance-model.configs[0].tensor-parallel.rank-endpoint-timeout-seconds=300");
        springArgs.addAll(List.of(args));
        return springArgs.toArray(String[]::new);
    }
}
