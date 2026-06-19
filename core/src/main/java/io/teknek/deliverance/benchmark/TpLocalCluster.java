package io.teknek.deliverance.benchmark;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.model.tensorparallel.GossipParallelMembership;
import io.teknek.deliverance.model.tensorparallel.GossipParallelSettings;
import io.teknek.deliverance.model.tensorparallel.TensorParallelDeploymentSpec;
import io.teknek.deliverance.model.tensorparallel.TensorParallelGenerationGroup;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.gossip.GossipSettings;
import io.teknek.gossip.Member;
import io.teknek.gossip.RemoteMember;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.URI;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;
import java.util.function.BooleanSupplier;

/**
 * Small command-line process for manually running tensor-parallel workers and a coordinator as separate JVMs.
 */
public final class TpLocalCluster {
    private static final Logger LOGGER = LoggerFactory.getLogger(TpLocalCluster.class);

    private TpLocalCluster() {
    }

    public static void main(String[] args) throws Exception {
        Options options = Options.parse(args);
        InferenceProfiler.setEnabled(options.profileStages);
        LOGGER.info("Starting TP local process role={} cluster={} node={} uri={} owner={} model={} deployment={} tpSize={} maxRanksPerWorker={}",
                options.role, options.cluster, options.nodeId, options.uri, options.owner, options.model,
                options.deploymentId, options.tensorParallelSize, options.maxRanksPerWorker);
        switch (options.role) {
            case WORKER -> runWorker(options);
            case COORDINATOR -> runCoordinator(options);
        }
    }

    private static void runWorker(Options options) throws InterruptedException {
        ModelFetcher fetcher = new ModelFetcher(options.owner, options.model);
        TensorParallelDeploymentSpec deploymentSpec = deploymentSpec(options);
        AutoModelForCausaLm.Builder builder = configuredBuilder(options, fetcher)
                .withParallelSettings(new GossipParallelSettings(options.cluster, options.nodeId, options.uri,
                        seedMembers(options), gossipSettings(), deploymentSpec, options.collectiveTransport));
        AbstractModel model = builder.buildAbstractModel();
        Runtime.getRuntime().addShutdownHook(new Thread(model::close, "tp-local-worker-shutdown-" + options.nodeId));
        LOGGER.info("Worker started node={} provider={} splitSize={} dtype={} workingDtype={} workingQtype={}",
                options.nodeId, model.getTensorProviderName(), model.getTensorProviderParallelSplitSize(),
                model.getModelDType(), model.getWorkingDType(), model.getWorkingQType());
        waitForever();
    }

    private static void runCoordinator(Options options) throws Exception {
        TensorParallelDeploymentSpec deploymentSpec = deploymentSpec(options);
        GossipParallelMembership membership = GossipParallelMembership.startObserver(new GossipParallelSettings(
                options.cluster, options.nodeId, options.uri, seedMembers(options), gossipSettings(), deploymentSpec,
                options.collectiveTransport));
        Runtime.getRuntime().addShutdownHook(new Thread(membership::close,
                "tp-local-coordinator-membership-shutdown-" + options.nodeId));

        eventually("candidates visible", () -> membership.candidateNodeIds().size() >= deploymentSpec.minimumPhysicalNodes(),
                options.readyTimeout);
        eventually("leader elected", () -> membership.electedLeader() != null, options.readyTimeout);
        eventually("assignment visible", () -> membership.findAssignment() != null, options.readyTimeout);
        LOGGER.info("Coordinator observed assignment leader={} assignment={} candidates={}", membership.electedLeader(),
                membership.findAssignment(), membership.candidateNodeIds());
        eventually("collective uri visible", () -> membership.findCollectiveUri() != null, options.readyTimeout);
        LOGGER.info("Coordinator observed collectiveUri={}", membership.findCollectiveUri());
        eventually("rank endpoints visible", () -> hasAllRankEndpoints(membership), options.rankEndpointTimeout);
        LOGGER.info("Coordinator observed rankEndpoints={}", membership.rankEndpointsForAssignment());

        ModelFetcher fetcher = new ModelFetcher(options.owner, options.model);
        AbstractModel coordinatorModel = configuredBuilder(options, fetcher).buildLocalTransformerModel();
        TensorParallelGenerationGroup group = membership.openGenerationGroup();
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            group.close();
            coordinatorModel.close();
        }, "tp-local-coordinator-shutdown-" + options.nodeId));
        LOGGER.info("Coordinator model ready provider={} splitSize={} dtype={} workingDtype={} workingQtype={}",
                coordinatorModel.getTensorProviderName(), coordinatorModel.getTensorProviderParallelSplitSize(),
                coordinatorModel.getModelDType(), coordinatorModel.getWorkingDType(), coordinatorModel.getWorkingQType());

        GeneratorParameters parameters = new GeneratorParameters()
                .withTemperature(options.temperature)
                .withMaxTokens(options.maxTokens);
        LOGGER.info("Coordinator starting probe generation promptChars={} maxTokens={} temperature={}",
                options.prompt.length(), options.maxTokens, options.temperature);
        Response response = group.generate(UUID.randomUUID(), coordinatorModel, PromptContext.of(options.prompt), parameters,
                new DoNothingGenerateEvent());
        LOGGER.info("Coordinator probe complete promptTokens={} generated={} totalMs={} finish={} response={}",
                response.promptTokens, response.generatedTokens.size(),
                String.format(Locale.ROOT, "%.1f", response.totalTimeMs), response.finishReason, response.responseText);
        InferenceProfiler.printSummary("tp-local-coordinator-probe", 40);
        waitForever();
    }

    private static AutoModelForCausaLm.Builder configuredBuilder(Options options, ModelFetcher fetcher) {
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(fetcher)
                .withWorkingMemoryType(options.workingDType)
                .withWorkingQuantType(options.workingQType)
                .withWrappedForkJoinPool(new WrappedForkJoinPool(new ForkJoinPool(options.poolSize,
                        ForkJoinPool.defaultForkJoinWorkerThreadFactory, null, true)));
        if (options.outputHeadQuantization != null) {
            builder.withOutputHeadQuantization(options.outputHeadQuantization);
        }
        return builder;
    }

    private static TensorParallelDeploymentSpec deploymentSpec(Options options) {
        return new TensorParallelDeploymentSpec(options.deploymentId, options.tensorParallelSize,
                options.maxRanksPerWorker);
    }

    private static GossipSettings gossipSettings() {
        GossipSettings settings = new GossipSettings();
        settings.setPersistRingState(false);
        settings.setPersistDataState(false);
        settings.setGossipInterval(100);
        settings.setCleanupInterval(2_000);
        return settings;
    }

    private static List<Member> seedMembers(Options options) {
        return options.seeds.stream()
                .map(seed -> new RemoteMember(options.cluster, seed.uri(), seed.nodeId()))
                .map(Member.class::cast)
                .toList();
    }

    private static boolean hasAllRankEndpoints(GossipParallelMembership membership) {
        try {
            membership.rankEndpointsForAssignment();
            return true;
        } catch (RuntimeException e) {
            return false;
        }
    }

    private static void eventually(String label, BooleanSupplier condition, Duration timeout) {
        LOGGER.info("Waiting for {} timeout={}", label, timeout);
        long deadline = System.nanoTime() + timeout.toNanos();
        while (System.nanoTime() < deadline) {
            if (condition.getAsBoolean()) {
                LOGGER.info("Ready {}", label);
                return;
            }
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("Interrupted waiting for " + label, e);
            }
        }
        throw new IllegalStateException("Timed out waiting for " + label + " after " + timeout);
    }

    private static void waitForever() throws InterruptedException {
        new CountDownLatch(1).await();
    }

    private enum Role {
        WORKER,
        COORDINATOR
    }

    private record Seed(String nodeId, URI uri) {
    }

    private static final class Options {
        private Role role;
        private String cluster = "deliverance-tp-local";
        private String nodeId;
        private URI uri;
        private final List<Seed> seeds = new ArrayList<>();
        private String deploymentId = "benchmark";
        private String collectiveTransport = "http";
        private int tensorParallelSize = 4;
        private int maxRanksPerWorker = 2;
        private String owner = "tjake";
        private String model = "gemma-2-2b-it-JQ4";
        private int poolSize = 16;
        private DType workingDType = DType.F32;
        private DType workingQType = DType.I8;
        private DType outputHeadQuantization = DType.Q4;
        private int maxTokens = 64;
        private float temperature = 0.0f;
        private boolean profileStages = true;
        private Duration readyTimeout = Duration.ofSeconds(120);
        private Duration rankEndpointTimeout = Duration.ofSeconds(300);
        private String prompt = "Explain tensor parallel inference in one short paragraph.";

        private static Options parse(String[] args) {
            Options options = new Options();
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--role" -> options.role = Role.valueOf(args[++i].toUpperCase(Locale.ROOT));
                    case "--cluster" -> options.cluster = args[++i];
                    case "--node-id" -> options.nodeId = args[++i];
                    case "--uri" -> options.uri = URI.create(args[++i]);
                    case "--seed" -> options.seeds.add(parseSeed(args[++i]));
                    case "--deployment" -> options.deploymentId = args[++i];
                    case "--collective-transport" -> options.collectiveTransport = args[++i].toLowerCase(Locale.ROOT);
                    case "--tensor-parallel-size" -> options.tensorParallelSize = Integer.parseInt(args[++i]);
                    case "--max-ranks-per-worker" -> options.maxRanksPerWorker = Integer.parseInt(args[++i]);
                    case "--owner" -> options.owner = args[++i];
                    case "--model" -> options.model = args[++i];
                    case "--pool-size" -> options.poolSize = Integer.parseInt(args[++i]);
                    case "--working-dtype" -> options.workingDType = DType.valueOf(args[++i]);
                    case "--working-qtype" -> options.workingQType = DType.valueOf(args[++i]);
                    case "--output-head-quantization" -> options.outputHeadQuantization = parseOptionalDType(args[++i]);
                    case "--max-tokens" -> options.maxTokens = Integer.parseInt(args[++i]);
                    case "--temperature" -> options.temperature = Float.parseFloat(args[++i]);
                    case "--profile-stages" -> options.profileStages = true;
                    case "--no-profile-stages" -> options.profileStages = false;
                    case "--ready-timeout-seconds" -> options.readyTimeout = Duration.ofSeconds(Long.parseLong(args[++i]));
                    case "--rank-endpoint-timeout-seconds" -> options.rankEndpointTimeout = Duration.ofSeconds(Long.parseLong(args[++i]));
                    case "--prompt" -> options.prompt = args[++i];
                    default -> throw new IllegalArgumentException("Unknown argument " + args[i]);
                }
            }
            Objects.requireNonNull(options.role, "--role is required");
            Objects.requireNonNull(options.nodeId, "--node-id is required");
            Objects.requireNonNull(options.uri, "--uri is required");
            if (options.seeds.isEmpty()) {
                throw new IllegalArgumentException("At least one --seed node=udp://host:port is required");
            }
            if (!options.collectiveTransport.equals("http") && !options.collectiveTransport.equals("netty")) {
                throw new IllegalArgumentException("--collective-transport must be http or netty");
            }
            return options;
        }

        private static Seed parseSeed(String raw) {
            int split = raw.indexOf('=');
            if (split <= 0 || split == raw.length() - 1) {
                throw new IllegalArgumentException("Seed must be nodeId=uri, got " + raw);
            }
            return new Seed(raw.substring(0, split), URI.create(raw.substring(split + 1)));
        }

        private static DType parseOptionalDType(String raw) {
            if (raw.equalsIgnoreCase("none") || raw.equalsIgnoreCase("off")) {
                return null;
            }
            return DType.valueOf(raw);
        }
    }
}
