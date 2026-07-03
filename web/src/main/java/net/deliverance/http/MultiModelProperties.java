package net.deliverance.http;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.CausalLanguageModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.model.tensorparallel.GossipParallelMembership;
import io.teknek.deliverance.model.tensorparallel.GossipParallelSettings;
import io.teknek.deliverance.model.tensorparallel.TensorParallelDeploymentSpec;
import io.teknek.deliverance.model.tensorparallel.TensorParallelGenerationGroup;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.gossip.GossipSettings;
import io.teknek.gossip.Member;
import io.teknek.gossip.RemoteMember;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.BooleanSupplier;

@Component
@ConfigurationProperties(prefix = "deliverance-model")
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
    private final TensorAllocator arrayQueueTensorAllocator;
    private final ConfigurableTensorProvider provider;
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

    public MultiModelConfiguration(MultiModelProperties multiModelProperties, MetricRegistry metricRegistry,
                                     TensorAllocator arrayQueueTensorAllocator,
                                     ConfigurableTensorProvider provider,
                                     WrappedForkJoinPool pool,
                                     @Value("${deliverance.kv.disk-dir:}") String kvDiskDirectory,
                                     @Value("${deliverance.kv.prefix.max-entries:10000}") int kvPrefixMaxEntries,
                                     @Value("${deliverance.kv.prefix.block-size:32}") int kvPrefixBlockSize,
                                     @Value("${deliverance.kv.prefix.max-tokens:512}") int kvPrefixMaxTokens,
                                     @Value("${deliverance.kv.prefix.max-checkpoints:4}") int kvPrefixMaxCheckpoints,
                                     @Value("${deliverance.kv.prefix.checkpoint-policy:START_AND_END}") String kvPrefixCheckpointPolicy,
                                     @Value("${deliverance.kv.prefix.compression:NONE}") String kvPrefixCompression,
                                     @Value("${deliverance.kv.prefix.turboquant.bits:4}") int kvPrefixTurboQuantBits,
                                     @Value("${deliverance.kv.context-rows-per-page-target:32}") int kvContextRowsPerPageTarget){
        this.multiModelProperties = multiModelProperties;
        this.metricRegistry = metricRegistry;
        this.arrayQueueTensorAllocator = arrayQueueTensorAllocator;
        this.provider = provider;
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
        File f = fetch.maybeDownload();
        return ModelSupport.loadEmbeddingModel(f, DType.F32, DType.I8, provider,
                metricRegistry, this.arrayQueueTensorAllocator, kvBufferCacheSettings());
    }

    private CausalLanguageModel causalLanguageModelFromConfig(MultiModelConfig config){
        ModelFetcher fetch = new ModelFetcher(config.getModelOwner(),config.getModelName());
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(fetch)
                .withMetricRegistry(metricRegistry)
                .withTensorAllocator(arrayQueueTensorAllocator)
                .withTensorProvider(provider)
                .withWrappedForkJoinPool(pool)
                .withKvBufferCacheSettings(kvBufferCacheSettings());
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
                tp.getCollectiveTransport()));
        eventually("tensor-parallel candidates visible", () -> membership.candidateNodeIds().size() >= deploymentSpec.minimumPhysicalNodes(),
                Duration.ofSeconds(tp.getReadyTimeoutSeconds()));
        eventually("tensor-parallel leader elected", () -> membership.electedLeader() != null,
                Duration.ofSeconds(tp.getReadyTimeoutSeconds()));
        eventually("tensor-parallel assignment visible", () -> membership.findAssignment() != null,
                Duration.ofSeconds(tp.getReadyTimeoutSeconds()));
        LOGGER.info("Spring coordinator observed tensor-parallel assignment model={} leader={} assignment={}",
                config.getModelName(), membership.electedLeader(), membership.findAssignment());
        eventually("tensor-parallel collective uri visible", () -> membership.findCollectiveUri() != null,
                Duration.ofSeconds(tp.getReadyTimeoutSeconds()));
        eventually("tensor-parallel rank endpoints visible", () -> hasAllRankEndpoints(membership),
                Duration.ofSeconds(tp.getRankEndpointTimeoutSeconds()));
        LOGGER.info("Spring coordinator observed tensor-parallel rank endpoints model={} endpoints={}",
                config.getModelName(), membership.rankEndpointsForAssignment());
        AbstractModel coordinatorModel = builder.buildLocalTransformerModel();
        TensorParallelGenerationGroup group = membership.openGenerationGroup();
        return new TensorParallelSpringCausalLanguageModel(coordinatorModel, group, membership);
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
        return settings;
    }
}

class TensorParallelSpringCausalLanguageModel implements CausalLanguageModel {
    private final AbstractModel coordinatorModel;
    private final TensorParallelGenerationGroup group;
    private final GossipParallelMembership membership;

    TensorParallelSpringCausalLanguageModel(AbstractModel coordinatorModel, TensorParallelGenerationGroup group,
            GossipParallelMembership membership) {
        this.coordinatorModel = coordinatorModel;
        this.group = group;
        this.membership = membership;
    }

    @Override
    public io.teknek.deliverance.generator.Response generate(UUID session,
            io.teknek.deliverance.safetensors.prompt.PromptContext promptContext,
            io.teknek.deliverance.generator.GeneratorParameters generatorParameters,
            io.teknek.deliverance.model.GenerateEvent onTokenWithTimings) {
        return group.generate(session, coordinatorModel, promptContext, generatorParameters, onTokenWithTimings);
    }

    @Override
    public io.teknek.deliverance.safetensors.Config getConfig() {
        return coordinatorModel.getConfig();
    }

    @Override
    public io.teknek.deliverance.grace.PreTrainedTokenizer getTokenizer() {
        return coordinatorModel.getTokenizer();
    }

    @Override
    public Optional<io.teknek.deliverance.safetensors.prompt.PromptSupport> promptSupport() {
        return coordinatorModel.promptSupport();
    }

    @Override
    public io.teknek.deliverance.toolcallparser.ToolCallParser getToolCallParser() {
        return coordinatorModel.getToolCallParser();
    }

    @Override
    public void close() throws IOException {
        try {
            group.close();
            coordinatorModel.close();
        } finally {
            membership.close();
        }
    }
}
