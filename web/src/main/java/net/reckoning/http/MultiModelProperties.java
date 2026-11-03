package net.reckoning.http;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.math.WrappedForkJoinPool;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.AutoModelForCausaLm;
import io.teknek.reckoning.model.CausalLanguageModel;
import io.teknek.reckoning.model.AutoModelForEmbeddings;
import io.teknek.reckoning.model.TensorProviderKind;
import io.teknek.reckoning.model.tensorparallel.GossipParallelMembership;
import io.teknek.reckoning.model.tensorparallel.GossipParallelSettings;
import io.teknek.reckoning.model.tensorparallel.TensorParallelAssignmentMode;
import io.teknek.reckoning.model.tensorparallel.TensorParallelDeploymentSpec;
import io.teknek.reckoning.model.tensorparallel.TensorParallelGenerationGroup;
import io.teknek.reckoning.model.tensorparallel.TensorParallelRankEndpoint;
import io.teknek.reckoning.model.tensorparallel.TensorParallelTimeoutSettings;
import io.teknek.reckoning.safetensors.fetch.ModelFetcher;
import io.teknek.reckoning.tensor.KvBufferCacheSettings;
import io.teknek.gossip.GossipSettings;
import io.teknek.gossip.Member;
import io.teknek.gossip.event.GossipState;
import io.teknek.gossip.RemoteMember;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.net.URI;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;



/*
@Component
@ConfigurationProperties(prefix = "reckoning-model")
public class MultiModelProperties {

    private List<MultiModelConfig> configs = new CopyOnWriteArrayList<>();

    public List<MultiModelConfig> getConfigs() {
        return configs;
    }

    public void setConfigs(List<MultiModelConfig> configs) {
        this.configs = configs;
    }

}

@Configuration
class MultiModelConfiguration {

    private static final Logger LOGGER = LoggerFactory.getLogger(MultiModelConfiguration.class);

    private final MultiModelProperties multiModelProperties;
    private final MetricRegistry metricRegistry;
    private final WrappedForkJoinPool pool;
    private final String kvDiskDirectory;
    private final int kvPrefixMaxEntries;
    private final int kvPrefixBlockSize;
    private final int kvPrefixMaxTokens;
    private final int kvPrefixMaxCheckpoints;
    private final String kvPrefixCheckpointPolicy;
    private final String kvPrefixCompression;
    private final int kvPrefixTurboQuantBits;
    private final int kvContextRowsPerPageTarget;
    private final String kvPrefixCacheMode;
    private final long kvSharedPrefixBlockCacheMaxBytes;
    private final String kvBlockStoragePolicy;
    private final int kvTurboQuantBits;
    private final String kvKeyDType;
    private final String kvValueDType;
    private final boolean kvSharedPrefixDiskCacheEnabled;
    private final String kvSharedPrefixDiskCachePath;
    private final long kvSharedPrefixDiskCacheMaxBytes;
    private final long kvSharedPrefixDiskCacheReservedFreeBytes;
    private final long kvSharedPrefixDiskCacheMinUsableBytes;
    private final int kvSharedPrefixDiskCacheAdmitMinTokens;
    private final int kvSharedPrefixDiskCacheWriterQueueSize;
    private final int groupedDecodeQkvSplitSize;
    private final boolean gpuPrefill;
    private final boolean gpuDecode;
    private final boolean gpuDecodeAttention;
    private final boolean gpuProvider;
    private final int simdParallelSplitSize;
    private final int maxPrefillBatchSize;

    public MultiModelConfiguration(MultiModelProperties multiModelProperties, MetricRegistry metricRegistry,
                                     WrappedForkJoinPool pool,
                                     @Value("${reckoning.kv.disk-dir:}") String kvDiskDirectory,
                                     @Value("${reckoning.kv.prefix.max-entries:10000}") int kvPrefixMaxEntries,
                                     @Value("${reckoning.kv.prefix.block-size:32}") int kvPrefixBlockSize,
                                     @Value("${reckoning.kv.prefix.max-tokens:512}") int kvPrefixMaxTokens,
                                     @Value("${reckoning.kv.prefix.max-checkpoints:4}") int kvPrefixMaxCheckpoints,
                                     @Value("${reckoning.kv.prefix.checkpoint-policy:START_AND_END}") String kvPrefixCheckpointPolicy,
                                      @Value("${reckoning.kv.prefix.compression:NONE}") String kvPrefixCompression,
                                      @Value("${reckoning.kv.prefix.turboquant.bits:4}") int kvPrefixTurboQuantBits,
                                      @Value("${reckoning.kv.context-rows-per-page-target:32}") int kvContextRowsPerPageTarget,
                                      @Value("${reckoning.kv.prefix.mode:SNAPSHOT}") String kvPrefixCacheMode,
                                      @Value("${reckoning.kv.shared-prefix.block-cache.max-bytes:536870912}") long kvSharedPrefixBlockCacheMaxBytes,
                                      @Value("${reckoning.kv.block-storage-policy:DENSE}") String kvBlockStoragePolicy,
                                      @Value("${reckoning.kv.turboquant.bits:4}") int kvTurboQuantBits,
                                      @Value("${reckoning.kv.key-dtype:F32}") String kvKeyDType,
                                      @Value("${reckoning.kv.value-dtype:F32}") String kvValueDType,
                                      @Value("${reckoning.kv.shared-prefix.disk-cache.enabled:false}") boolean kvSharedPrefixDiskCacheEnabled,
                                      @Value("${reckoning.kv.shared-prefix.disk-cache.path:}") String kvSharedPrefixDiskCachePath,
                                      @Value("${reckoning.kv.shared-prefix.disk-cache.max-bytes:2147483648}") long kvSharedPrefixDiskCacheMaxBytes,
                                      @Value("${reckoning.kv.shared-prefix.disk-cache.reserved-free-bytes:1073741824}") long kvSharedPrefixDiskCacheReservedFreeBytes,
                                      @Value("${reckoning.kv.shared-prefix.disk-cache.min-usable-bytes:1073741824}") long kvSharedPrefixDiskCacheMinUsableBytes,
                                      @Value("${reckoning.kv.shared-prefix.disk-cache.admit-min-tokens:256}") int kvSharedPrefixDiskCacheAdmitMinTokens,
                                      @Value("${reckoning.kv.shared-prefix.disk-cache.writer-queue-size:128}") int kvSharedPrefixDiskCacheWriterQueueSize,
                                      @Value("${reckoning.grouped-decode-qkv-split-size:0}") int groupedDecodeQkvSplitSize,
                                      @Value("${reckoning.gpu.prefill:false}") boolean gpuPrefill,
                                       @Value("${reckoning.gpu.decode:false}") boolean gpuDecode,
                                       @Value("${reckoning.gpu.decode-attention:false}") boolean gpuDecodeAttention,
                                       @Value("${reckoning.gpu.provider:true}") boolean gpuProvider,
                                       @Value("${reckoning.max-prefill-batch-size:0}") int maxPrefillBatchSize,
                                       @Value("${reckoning.tensor.operations.simd.parallel-split-size:0}") int simdParallelSplitSize){
        this.multiModelProperties = multiModelProperties;
        this.metricRegistry = metricRegistry;
        this.pool = pool;
        this.kvDiskDirectory = kvDiskDirectory;
        this.kvPrefixMaxEntries = kvPrefixMaxEntries;
        this.kvPrefixBlockSize = kvPrefixBlockSize;
        this.kvPrefixMaxTokens = kvPrefixMaxTokens;
        this.kvPrefixMaxCheckpoints = kvPrefixMaxCheckpoints;
        this.kvPrefixCheckpointPolicy = kvPrefixCheckpointPolicy;
        this.kvPrefixCompression = kvPrefixCompression;
        this.kvPrefixTurboQuantBits = kvPrefixTurboQuantBits;
        this.kvContextRowsPerPageTarget = kvContextRowsPerPageTarget;
        this.kvPrefixCacheMode = kvPrefixCacheMode;
        this.kvSharedPrefixBlockCacheMaxBytes = kvSharedPrefixBlockCacheMaxBytes;
        this.kvBlockStoragePolicy = kvBlockStoragePolicy;
        this.kvTurboQuantBits = kvTurboQuantBits;
        this.kvKeyDType = kvKeyDType;
        this.kvValueDType = kvValueDType;
        this.kvSharedPrefixDiskCacheEnabled = kvSharedPrefixDiskCacheEnabled;
        this.kvSharedPrefixDiskCachePath = kvSharedPrefixDiskCachePath;
        this.kvSharedPrefixDiskCacheMaxBytes = kvSharedPrefixDiskCacheMaxBytes;
        this.kvSharedPrefixDiskCacheReservedFreeBytes = kvSharedPrefixDiskCacheReservedFreeBytes;
        this.kvSharedPrefixDiskCacheMinUsableBytes = kvSharedPrefixDiskCacheMinUsableBytes;
        this.kvSharedPrefixDiskCacheAdmitMinTokens = kvSharedPrefixDiskCacheAdmitMinTokens;
        this.kvSharedPrefixDiskCacheWriterQueueSize = kvSharedPrefixDiskCacheWriterQueueSize;
        this.groupedDecodeQkvSplitSize = groupedDecodeQkvSplitSize;
        this.gpuPrefill = gpuPrefill;
        this.gpuDecode = gpuDecode;
        this.gpuDecodeAttention = gpuDecodeAttention;
        this.gpuProvider = gpuProvider;
        this.maxPrefillBatchSize = maxPrefillBatchSize;
        this.simdParallelSplitSize = simdParallelSplitSize;
    }


    @Bean
    public Map<MultiModelConfig, CausalLanguageModel> causalLanguageModels(){
        Map<MultiModelConfig, CausalLanguageModel> models = new HashMap<>();
        for (var x : multiModelProperties.getConfigs()){
            if ("GENERATION".equalsIgnoreCase(x.getInferenceType())) {
                models.put(x, causalLanguageModelFromConfig(x));
            }
        }
        return models;
    }

    @Bean
    public Map<MultiModelConfig, AbstractModel> embeddingModels(){
        Map<MultiModelConfig, AbstractModel> models = new HashMap<>();
        for (var x : multiModelProperties.getConfigs()){
            if ("EMBEDDING".equalsIgnoreCase(x.getInferenceType())) {
                models.put(x, embeddingModelFromConfig(x));
            }
        }
        return models;
    }

    private AbstractModel embeddingModelFromConfig(MultiModelConfig config){
        ModelFetcher fetch = new ModelFetcher(config.getModelOwner(),config.getModelName());
        AutoModelForEmbeddings.Builder embeddingBuilder = AutoModelForEmbeddings.newBuilder(fetch);
        embeddingBuilder.withMetricRegistry(metricRegistry);
        embeddingBuilder.withWrappedForkJoinPool(pool);
        embeddingBuilder.withKvBufferCacheSettings(kvBufferCacheSettings());
        return embeddingBuilder.buildLocalEmbeddingModel();
    }

    private CausalLanguageModel causalLanguageModelFromConfig(MultiModelConfig config){
        ModelFetcher fetch = new ModelFetcher(config.getModelOwner(), config.getModelName());
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(fetch)
                .withMetricRegistry(metricRegistry)
                .withWrappedForkJoinPool(pool)
                .withKvBufferCacheSettings(kvBufferCacheSettings());
        if (groupedDecodeQkvSplitSize > 0) {
            builder.withGroupedDecodeQkvSplitSize(groupedDecodeQkvSplitSize);
        }
        if (maxPrefillBatchSize > 0) {
            builder.withMaxPrefillBatchSize(maxPrefillBatchSize);
        }

        builder.withGpuPrefill(gpuPrefill)
                .withGpuDecode(gpuDecode)

                .withGpuDecodeAttention(gpuDecodeAttention)
                .withGpuProvider(gpuProvider);
        if (simdParallelSplitSize > 0) {
            builder.withParallelSplitSizeFixed(TensorProviderKind.SIMD, simdParallelSplitSize);
        }
        if (config.getOutputHeadQuantization() != null
                && !config.getOutputHeadQuantization().isBlank()
                && !"none".equalsIgnoreCase(config.getOutputHeadQuantization())
                && !"off".equalsIgnoreCase(config.getOutputHeadQuantization())) {
            builder.withOutputHeadQuantization(DType.valueOf(config.getOutputHeadQuantization()));
        }
        if (config.getTensorParallel() != null && config.getTensorParallel().isEnabled()) {
            return tensorParallelCausalLanguageModel(config, builder);
        }
        return builder.build();
    }

    private CausalLanguageModel tensorParallelCausalLanguageModel(MultiModelConfig config,
            AutoModelForCausaLm.Builder builder) {
        MultiModelConfig.TensorParallelConfig tp = config.getTensorParallel();
        if (tp.getOutputHeadQuantization() != null
                && !tp.getOutputHeadQuantization().isBlank()
                && !"none".equalsIgnoreCase(tp.getOutputHeadQuantization())
                && !"off".equalsIgnoreCase(tp.getOutputHeadQuantization())) {
            builder.withOutputHeadQuantization(DType.valueOf(tp.getOutputHeadQuantization()));
        }
        TensorParallelDeploymentSpec deploymentSpec = new TensorParallelDeploymentSpec(tp.getDeployment(),
                tp.getSize(), tp.getMaxRanksPerWorker());
        GossipParallelMembership membership = GossipParallelMembership.startObserver(new GossipParallelSettings(
                tp.getCluster(), tp.getNodeId(), URI.create(tp.getUri()), seedMembers(tp), gossipSettings(), deploymentSpec,
                tp.getCollectiveTransport(), TensorParallelTimeoutSettings.ofSeconds(
                tp.getRankConnectTimeoutSeconds(), tp.getRankRequestTimeoutSeconds(),
                tp.getRankOperationTimeoutSeconds(), tp.getRankCloseTimeoutSeconds()),
                TensorParallelAssignmentMode.fromString(tp.getAssignmentMode())));
        AbstractModel coordinatorModel = builder.buildLocalTransformerModel();
        LOGGER.info("Spring tensor-parallel coordinator model loaded model={} deployment={} size={} initialState=NOT_READY",
                config.getModelName(), tp.getDeployment(), tp.getSize());
        return new TensorParallelSpringCausalLanguageModel(coordinatorModel, null, membership);
    }

    private static List<Member> seedMembers(MultiModelConfig.TensorParallelConfig config) {
        List<Member> members = new ArrayList<>();
        for (String raw : config.getSeeds()) {
            int split = raw.indexOf('=');
            if (split <= 0 || split == raw.length() - 1) {
                throw new IllegalArgumentException("Tensor-parallel seed must be nodeId=uri, got " + raw);
            }
            members.add(new RemoteMember(config.getCluster(), URI.create(raw.substring(split + 1)), raw.substring(0, split)));
        }
        return members;
    }

    private static GossipSettings gossipSettings() {
        GossipSettings settings = new GossipSettings();
        settings.setPersistRingState(false);
        settings.setPersistDataState(false);
        settings.setGossipInterval(100);
        settings.setCleanupInterval(2_000);
        return settings;
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

    private KvBufferCacheSettings kvBufferCacheSettings() {
        KvBufferCacheSettings settings;
        if (kvDiskDirectory == null || kvDiskDirectory.isBlank()) {
            settings = new KvBufferCacheSettings(true);
        } else {
            settings = new KvBufferCacheSettings(new File(kvDiskDirectory));
        }
        settings.setMaxEntries(kvPrefixMaxEntries);
        settings.setBlockSize(kvPrefixBlockSize);
        settings.setMaxPrefixTokensPerPrompt(kvPrefixMaxTokens);
        settings.setMaxPrefixCheckpointsPerPrompt(kvPrefixMaxCheckpoints);
        settings.setPrefixCheckpointPolicy(KvBufferCacheSettings.PrefixCheckpointPolicy.valueOf(kvPrefixCheckpointPolicy));
        settings.setPrefixCompression(KvBufferCacheSettings.PrefixCompression.valueOf(kvPrefixCompression));
        settings.setPrefixTurboQuantBits(kvPrefixTurboQuantBits);
        settings.setContextRowsPerPageTarget(kvContextRowsPerPageTarget);
        settings.setPrefixCacheMode(KvBufferCacheSettings.PrefixCacheMode.valueOf(kvPrefixCacheMode));
        settings.setSharedPrefixBlockCacheMaxBytes(kvSharedPrefixBlockCacheMaxBytes);
        settings.setKvBlockStoragePolicy(KvBufferCacheSettings.KvBlockStoragePolicy.valueOf(kvBlockStoragePolicy));
        settings.setKvTurboQuantBits(kvTurboQuantBits);
        settings.setKvKeyDType(DType.valueOf(kvKeyDType));
        settings.setKvValueDType(DType.valueOf(kvValueDType));
        settings.setSharedPrefixDiskCacheEnabled(kvSharedPrefixDiskCacheEnabled);
        if (kvSharedPrefixDiskCachePath != null && !kvSharedPrefixDiskCachePath.isBlank()) {
            settings.setSharedPrefixDiskCachePath(new File(kvSharedPrefixDiskCachePath));
        }
        settings.setSharedPrefixDiskCacheMaxBytes(kvSharedPrefixDiskCacheMaxBytes);
        settings.setSharedPrefixDiskCacheReservedFreeBytes(kvSharedPrefixDiskCacheReservedFreeBytes);
        settings.setSharedPrefixDiskCacheMinUsableBytes(kvSharedPrefixDiskCacheMinUsableBytes);
        settings.setSharedPrefixDiskCacheAdmitMinTokens(kvSharedPrefixDiskCacheAdmitMinTokens);
        settings.setSharedPrefixDiskCacheWriterQueueSize(kvSharedPrefixDiskCacheWriterQueueSize);
        return settings;
    }
}
*/
class TensorParallelSpringCausalLanguageModel implements CausalLanguageModel {
    private static final Logger LOGGER = LoggerFactory.getLogger(TensorParallelSpringCausalLanguageModel.class);
    private static final long RECONCILE_INTERVAL_MILLIS = 500;
    private final AbstractModel coordinatorModel;
    private final AtomicReference<TensorParallelGenerationGroup> group;
    private final GossipParallelMembership membership;
    private final AtomicReference<Readiness> readiness;
    private final AtomicInteger activeGenerations = new AtomicInteger();
    private final ScheduledExecutorService readinessReconciler;
    private final Object groupLock = new Object();

    TensorParallelSpringCausalLanguageModel(AbstractModel coordinatorModel, TensorParallelGenerationGroup group,
            GossipParallelMembership membership) {
        this(coordinatorModel, group, membership, true);
    }

    TensorParallelSpringCausalLanguageModel(AbstractModel coordinatorModel, TensorParallelGenerationGroup group,
            GossipParallelMembership membership, boolean startReadinessReconciler) {
        this.coordinatorModel = coordinatorModel;
        this.group = new AtomicReference<>(group);
        this.membership = membership;
        this.readiness = new AtomicReference<>(group == null
                ? Readiness.notReady("generation group is not open")
                : Readiness.ready());
        if (startReadinessReconciler) {
            this.readinessReconciler = Executors.newSingleThreadScheduledExecutor(task -> {
                Thread thread = new Thread(task, "reckoning-tp-readiness-" + membership.localNodeId());
                thread.setDaemon(true);
                return thread;
            });
            this.readinessReconciler.scheduleWithFixedDelay(this::reconcileReadinessSafely,
                    RECONCILE_INTERVAL_MILLIS, RECONCILE_INTERVAL_MILLIS, TimeUnit.MILLISECONDS);
        } else {
            this.readinessReconciler = null;
        }
        LOGGER.info("Tensor-parallel web coordinator registered gossip listener node={} assignment={}",
                membership.localNodeId(), membership.findAssignment());
    }

    private void reconcileReadinessSafely() {
        try {
            reconcileReadiness();
        } catch (RuntimeException e) {
            readiness.set(Readiness.notReady(e.getMessage()));
            LOGGER.warn("Tensor-parallel readiness reconciliation failed", e);
        }
    }

    void reconcileReadiness() {
        synchronized (groupLock) {
            ReadinessCheck check = checkReadiness(null);
            if (!check.ready()) {
                readiness.set(Readiness.notReady(check.diagnostics()));
                TensorParallelGenerationGroup current = group.get();
                if (current != null && activeGenerations.get() == 0) {
                    LOGGER.warn("Invalidating tensor-parallel generation group: {}", check.diagnostics());
                    group.set(null);
                    current.close();
                } else if (current != null) {
                    LOGGER.info("Leaving tensor-parallel generation group open until active requests finish diagnostics={} activeGenerations={}",
                            check.diagnostics(), activeGenerations.get());
                }
                return;
            }
            if (group.get() != null) {
                readiness.set(Readiness.ready());
                return;
            }
            TensorParallelGenerationGroup reopened = membership.openGenerationGroup();
            group.set(reopened);
            readiness.set(Readiness.ready());
            LOGGER.info("Reactivated tensor-parallel generation group assignmentHash={} endpoints={}",
                    check.assignmentHash(), check.endpoints());
        }
    }

    @Override
    public io.teknek.reckoning.generator.Response generate(UUID session,
            io.teknek.reckoning.safetensors.prompt.PromptContext promptContext,
            io.teknek.reckoning.generator.GeneratorParameters generatorParameters,
            io.teknek.reckoning.model.GenerateEvent onTokenWithTimings) {
        TensorParallelGenerationGroup current;
        synchronized (groupLock) {
            current = assertTensorParallelReady();
            activeGenerations.incrementAndGet();
        }
        try {
            return current.generate(session, coordinatorModel, promptContext, generatorParameters, onTokenWithTimings);
        } finally {
            activeGenerations.decrementAndGet();
        }
    }

    private TensorParallelGenerationGroup assertTensorParallelReady() {
        Readiness current = readiness.get();
        if (current.state() == ReadinessState.NOT_READY) {
            LOGGER.info("Rejecting tensor-parallel request because readiness state is {} diagnostics={}",
                    current.state(), current.diagnostics());
            throw notReady(current.diagnostics());
        }
        TensorParallelGenerationGroup currentGroup = group.get();
        if (currentGroup == null) {
            LOGGER.info("Rejecting tensor-parallel request because generation group is not open");
            throw notReady("generation group is not open");
        }
        ReadinessCheck check = checkReadiness(null);
        if (!check.ready()) {
            LOGGER.info("Rejecting tensor-parallel request because {}", check.diagnostics());
            throw notReady(check.diagnostics());
        }
        LOGGER.info("Tensor-parallel request readiness OK assignmentHash={} endpoints={}",
                check.assignmentHash(), check.endpoints());
        return currentGroup;
    }

    private ReadinessCheck checkReadiness(String additionallyLiveNodeId) {
        var assignment = membership.findAssignment();
        if (assignment == null) {
            return ReadinessCheck.notReady("assignment is not visible in gossip");
        }
        if (membership.findCollectiveUri() == null) {
            return ReadinessCheck.notReady("collective URI is not visible in gossip");
        }
        List<TensorParallelRankEndpoint> endpoints;
        try {
            endpoints = membership.rankEndpointsForAssignment();
        } catch (RuntimeException e) {
            return ReadinessCheck.notReady(e.getMessage());
        }
        Set<Integer> ranks = new HashSet<>();
        List<Integer> duplicateRanks = new ArrayList<>();
        List<Integer> missingRanks = new ArrayList<>();
        for (var endpoint : endpoints) {
            if (!ranks.add(endpoint.rank())) {
                duplicateRanks.add(endpoint.rank());
            }
            if (endpoint.uri() == null || endpoint.uri().isBlank()) {
                return ReadinessCheck.notReady("rank " + endpoint.rank() + " has no endpoint URI");
            }
        }
        for (int rank = 0; rank < assignment.tensorParallelSize(); rank++) {
            if (!ranks.contains(rank)) {
                missingRanks.add(rank);
            }
        }
        if (!missingRanks.isEmpty() || !duplicateRanks.isEmpty() || endpoints.size() != assignment.tensorParallelSize()) {
            return ReadinessCheck.notReady("expected ranks 0.." + (assignment.tensorParallelSize() - 1)
                    + " but found endpoints=" + endpoints
                    + " missing=" + missingRanks
                    + " duplicates=" + duplicateRanks);
        }
        Set<String> liveNodeIds = new HashSet<>();
        liveNodeIds.add(membership.localNodeId());
        if (additionallyLiveNodeId != null) {
            liveNodeIds.add(additionallyLiveNodeId);
        }
        for (var member : membership.liveMembers()) {
            liveNodeIds.add(member.getId());
        }
        List<String> downOwners = endpoints.stream()
                .map(TensorParallelRankEndpoint::nodeId)
                .distinct()
                .filter(nodeId -> !liveNodeIds.contains(nodeId))
                .toList();
        if (!downOwners.isEmpty()) {
            return ReadinessCheck.notReady("gossip reports rank owner nodes down: " + downOwners);
        }
        return ReadinessCheck.ready(assignment.assignmentHash(), endpoints);
    }

    private ResponseStatusException notReady(String reason) {
        return new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,
                "Tensor-parallel deployment is not ready: " + reason);
    }

    Map<String, Object> diagnostics() {
        Readiness currentReadiness = readiness.get();
        TensorParallelGenerationGroup currentGroup = group.get();
        Map<String, Object> membershipDiagnostics = membership.diagnostics();
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("state", currentReadiness.state().name());
        result.put("diagnostics", currentReadiness.diagnostics());
        result.put("groupOpen", currentGroup != null);
        result.put("activeGenerations", activeGenerations.get());
        result.put("membership", membershipDiagnostics);
        return result;
    }

    Map<String, Object> gossipDiagnostics() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("nodeId", membership.localNodeId());
        result.put("liveMembers", membership.liveMembers().stream().map(Member::getId).sorted().toList());
        result.put("candidates", membership.candidateNodeIds());
        result.put("leader", membership.electedLeader() == null ? "" : membership.electedLeader());
        result.put("assignment", membership.findAssignment() == null ? "" : membership.findAssignment().toString());
        return result;
    }

    Map<String, Object> endpointDiagnostics() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("nodeId", membership.localNodeId());
        result.put("collectiveUri", membership.findCollectiveUri() == null ? "" : membership.findCollectiveUri().toString());
        try {
            result.put("rankEndpoints", membership.rankEndpointsForAssignment());
        } catch (RuntimeException e) {
            result.put("rankEndpoints", List.of());
            result.put("error", e.getMessage());
        }
        return result;
    }

    Map<String, Object> assignRank(String nodeId, int rank) {
        membership.assignRankManually(nodeId, rank);
        return assignmentDiagnostics();
    }

    private Map<String, Object> assignmentDiagnostics() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("nodeId", membership.localNodeId());
        result.put("manualAssignment", membership.findManualAssignment().ranks());
        result.put("assignment", membership.findAssignment() == null ? "" : membership.findAssignment());
        result.put("collectiveUri", membership.findCollectiveUri() == null ? "" : membership.findCollectiveUri().toString());
        return result;
    }

    enum ReadinessState {
        READY,
        NOT_READY
    }

    record Readiness(ReadinessState state, String diagnostics) {
        static Readiness ready() {
            return new Readiness(ReadinessState.READY, "ready");
        }

        static Readiness notReady(String diagnostics) {
            return new Readiness(ReadinessState.NOT_READY, diagnostics);
        }
    }

    record ReadinessCheck(boolean ready, String diagnostics, String assignmentHash,
            List<TensorParallelRankEndpoint> endpoints) {
        static ReadinessCheck ready(String assignmentHash, List<TensorParallelRankEndpoint> endpoints) {
            return new ReadinessCheck(true, "ready", assignmentHash, endpoints);
        }

        static ReadinessCheck notReady(String diagnostics) {
            return new ReadinessCheck(false, diagnostics, null, List.of());
        }
    }

    @Override
    public io.teknek.reckoning.safetensors.Config getConfig() {
        return coordinatorModel.getConfig();
    }

    @Override
    public io.teknek.reckoning.grace.PreTrainedTokenizer getTokenizer() {
        return coordinatorModel.getTokenizer();
    }

    @Override
    public Optional<io.teknek.reckoning.safetensors.prompt.PromptSupport> promptSupport() {
        return coordinatorModel.promptSupport();
    }

    @Override
    public io.teknek.reckoning.toolcallparser.ToolCallParser getToolCallParser() {
        return coordinatorModel.getToolCallParser();
    }

    @Override
    public void registerLoraAdapter(String adapterId, io.teknek.reckoning.safetensors.LoraAdapter adapter) {
        coordinatorModel.registerLoraAdapter(adapterId, adapter);
    }

    @Override
    public void registerLoraAdapter(String adapterId, io.teknek.reckoning.safetensors.fetch.LoraAdapterModelFetcher fetcher) {
        coordinatorModel.registerLoraAdapter(adapterId, fetcher);
    }

    @Override
    public void unregisterLoraAdapter(String adapterId) {
        coordinatorModel.unregisterLoraAdapter(adapterId);
    }

    @Override
    public void setActiveAdapter(String adapterId) {
        coordinatorModel.setActiveAdapter(adapterId);
    }

    @Override
    public void clearActiveAdapter() {
        coordinatorModel.clearActiveAdapter();
    }

    @Override
    public void close() throws IOException {
        try {
            if (readinessReconciler != null) {
                readinessReconciler.shutdownNow();
            }
            synchronized (groupLock) {
                TensorParallelGenerationGroup current = group.getAndSet(null);
                if (current != null) {
                    current.close();
                }
            }
            coordinatorModel.close();
        } finally {
            membership.close();
        }
    }
}
