package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.SingleRankTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.GossipParallelMembership;
import io.teknek.deliverance.model.tensorparallel.GossipParallelSettings;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.ModelQuantizer;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import io.teknek.deliverance.toolcallparser.LlamaToolCallParser;
import io.teknek.deliverance.toolcallparser.QwenToolCallParser;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.lang.reflect.InvocationTargetException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;
import java.util.function.Function;

public class AutoModelForCausaLm {
    private static final Logger LOGGER = LoggerFactory.getLogger(AutoModelForCausaLm.class);
    public static void applyTuning(ModelFetcher fetcher, Builder b){
        if (fetcher.getName().startsWith("Llama")){
            b.withToolCallParser(new LlamaToolCallParser());
        }
        if (fetcher.getName().startsWith("Qwen")){
            b.withToolCallParser(new QwenToolCallParser());
        }
    }

    public static CausalLanguageModel fromPretrained(ModelFetcher fetcher){
        Builder b = new Builder(fetcher);
        applyTuning(fetcher, b);
        return b.build();
    }

    public static Builder newBuilder(ModelFetcher fetcher){
        //There is an argument to be made we shouldnt tune both sides
        Builder b = new Builder(fetcher);
        applyTuning(fetcher, b);
        return b;
    }


    public static class Builder {
        private final ModelFetcher fetch;
        private MetricRegistry mr = new MetricRegistry();
        private TensorAllocator allocator = new ArrayQueueTensorAllocator(mr);
        private DType workingMem = DType.F32;
        private DType workingQuant = DType.I8;
        private ToolCallParser toolCallParser = new DefaultToolCallParser();

        private KvBufferCacheSettings settings = new KvBufferCacheSettings(true);
        private ConfigurableTensorProvider provider;
        private WrappedForkJoinPool pool;
        private String oobCheck = "2";
        private TensorParallelContext tensorParallelContext = new StaticTensorParallelContext(0, 1);
        private TensorParallelCollectives tensorParallelCollectives = new SingleRankTensorParallelCollectives();
        private Optional<GossipParallelSettings> parallelSettings = Optional.empty();
        private Optional<DType> outputHeadQuantization = Optional.empty();
        private Optional<QuantizeOnDemand> quantizeOnDemand = Optional.empty();
        private boolean download = true;

        record QuantizeOnDemand(DType targetType, String outputOwner, String outputModel) {
            QuantizeOnDemand {
                Objects.requireNonNull(targetType, "targetType");
                Objects.requireNonNull(outputOwner, "outputOwner");
                Objects.requireNonNull(outputModel, "outputModel");
                if (outputOwner.isBlank() || outputModel.isBlank()) {
                    throw new IllegalArgumentException("outputOwner and outputModel must not be blank");
                }
            }
        }

        public Builder(ModelFetcher fetch){
            this.fetch = fetch;
        }

        public Builder withMetricRegistry(MetricRegistry metricRegistry){
            mr = metricRegistry;
            return this;
        }
        public Builder withTensorAllocator(TensorAllocator tensorAllocator){
            this.allocator = tensorAllocator;
            return this;
        }
        public Builder withKvBufferCacheSettings(KvBufferCacheSettings settings){
            this.settings = settings;
            return this;
        }
        public Builder withWorkingMemoryType(DType type){
            this.workingMem = type;
            return this;
        }
        public Builder withWorkingQuantType(DType type){
            this.workingQuant = type;
            return this;
        }
        public Builder withTensorProvider(ConfigurableTensorProvider provider){
            this.provider = provider;
            return this;
        }
        public Builder withToolCallParser(ToolCallParser toolCallParser){
            this.toolCallParser = toolCallParser;
            return this;
        }
        public Builder withWrappedForkJoinPool(WrappedForkJoinPool pool){
            this.pool = pool;
            return this;
        }
        public Builder withTensorParallelContext(TensorParallelContext tensorParallelContext) {
            this.tensorParallelContext = Objects.requireNonNull(tensorParallelContext, "tensorParallelContext");
            return this;
        }
        public Builder withTensorParallel(int rank, int size) {
            return withTensorParallelContext(new StaticTensorParallelContext(rank, size));
        }
        public Builder withTensorParallelCollectives(TensorParallelCollectives tensorParallelCollectives) {
            this.tensorParallelCollectives = Objects.requireNonNull(tensorParallelCollectives, "tensorParallelCollectives");
            return this;
        }
        /**
         * Enables gossip-coordinated tensor-parallel runtime construction for {@link #build()}.
         *
         * <p>The returned {@link CausalLanguageModel} keeps the same public generation API, but it is not behaviorally
         * identical to a single local model in every respect. Current tensor-parallel generation uses rank-local KV state
         * for each request and does not expose local prefix-cache reuse; numerical output equivalence is model-family and
         * tensor-provider dependent. Gemma2 is the primary tested tensor-parallel family.</p>
         */
        public Builder withParallelSettings(GossipParallelSettings parallelSettings) {
            this.parallelSettings = Optional.of(Objects.requireNonNull(parallelSettings, "parallelSettings"));
            return this;
        }

        /**
         * Requests a specific dtype for causal-LM output head weights.
         *
         * <p>The output head projects the final hidden state to vocabulary logits on every generated token, so its dtype
         * can materially affect generation throughput. Some quantized models keep embedding/lm-head tensors dense for
         * quality; this option lets callers explicitly test or choose a quantized output head, for example {@code Q4},
         * without changing the rest of the model loading policy.</p>
         *
         * <p>This is opt-in because it directly changes logits and can change generated tokens. Callers should validate
         * first-token/top-k parity or run model-specific golden prompts before using it as a default.</p>
         */
        public Builder withOutputHeadQuantization(DType outputHeadQuantization) {
            this.outputHeadQuantization = Optional.of(Objects.requireNonNull(outputHeadQuantization, "outputHeadQuantization"));
            return this;
        }

        /**
         * Controls whether missing or incomplete model files may be downloaded. Defaults to {@code true}.
         * Set to {@code false} to require the model to already exist in the local Deliverance cache.
         */
        public Builder withDownload(boolean download) {
            this.download = download;
            return this;
        }

        /**
         * Loads a cached quantized target when it exists, otherwise quantizes the source model into
         * the local Deliverance cache and loads that generated target. Source download behavior is
         * still controlled by {@link #withDownload(boolean)}.
         */
        public Builder withQuantizeOnDemand(DType targetType, String outputOwner, String outputModel) {
            this.quantizeOnDemand = Optional.of(new QuantizeOnDemand(targetType, outputOwner, outputModel));
            return this;
        }
        public GossipParallelMembership startParallelMembership() {
            return GossipParallelMembership.start(parallelSettings.orElseThrow(() ->
                    new IllegalStateException("parallelSettings must be configured before starting membership")));
        }

        /**
         * Creates one builder per tensor-parallel rank assigned to this physical node.
         *
         * <p>The assignment comes from gossip membership. This method does not build or load models; it only projects the
         * committed rank assignment into rank-specific builders.</p>
         */
        public List<Builder> localAssignedRankBuilders(GossipParallelMembership membership) {
            return localAssignedRankBuilders(membership, ignored -> tensorParallelCollectives);
        }

        public List<Builder> localAssignedRankBuilders(GossipParallelMembership membership,
                Function<TensorParallelContext, TensorParallelCollectives> collectivesFactory) {
            Objects.requireNonNull(membership, "membership");
            Objects.requireNonNull(collectivesFactory, "collectivesFactory");
            if (!membership.assignmentMatchesLocalTopology()) {
                throw new IllegalStateException("Committed tensor-parallel assignment does not match local topology");
            }
            int tensorParallelSize = membership.requireAssignment().tensorParallelSize();
            List<Builder> builders = new ArrayList<>();
            for (int rank : membership.localRanks()) {
                TensorParallelContext context = new StaticTensorParallelContext(rank, tensorParallelSize);
                builders.add(copyForRank(context, collectivesFactory.apply(context)));
            }
            return List.copyOf(builders);
        }

        public List<AbstractModel> buildLocalAssignedRanks(GossipParallelMembership membership) {
            return buildLocalAssignedRanks(membership, ignored -> tensorParallelCollectives);
        }

        public List<AbstractModel> buildLocalAssignedRanks(GossipParallelMembership membership,
                Function<TensorParallelContext, TensorParallelCollectives> collectivesFactory) {
            List<AbstractModel> models = new ArrayList<>();
            for (Builder builder : localAssignedRankBuilders(membership, collectivesFactory)) {
                models.add(builder.buildLocalTransformerModel());
            }
            return List.copyOf(models);
        }

        private Builder copyForRank(TensorParallelContext context, TensorParallelCollectives collectives) {
            Builder copy = new Builder(fetch);
            copy.mr = this.mr;
            copy.allocator = this.allocator;
            copy.workingMem = this.workingMem;
            copy.workingQuant = this.workingQuant;
            copy.toolCallParser = this.toolCallParser;
            copy.settings = this.settings;
            copy.provider = this.provider;
            copy.pool = this.pool;
            copy.oobCheck = this.oobCheck;
            copy.tensorParallelContext = context;
            copy.tensorParallelCollectives = Objects.requireNonNull(collectives, "collectives");
            copy.outputHeadQuantization = this.outputHeadQuantization;
            copy.quantizeOnDemand = this.quantizeOnDemand;
            copy.download = this.download;
            return copy;
        }
        /** This is a JVM wide property! **/
        public Builder withSystemPropertyForVectorOobCheck(String value){
            this.oobCheck = value;
            return this;
        }

        public CausalLanguageModel build(){
            AbstractModel model = loadLocalTransformerModel();
            if (parallelSettings.isPresent()) {
                GossipParallelMembership membership = GossipParallelMembership.start(parallelSettings.get());
                model.setGossipParallelMembership(membership);
                membership.startWorkerWhenReady(this);
            }
            return DefaultCausalLanguageModel.local(model);
        }

        /**
         * Builds the local transformer executor used by tests, tensor-parallel rank workers, and migration code.
         * Prefer {@link #build()} for user-facing causal language model loading.
         */
        public AbstractModel buildLocalTransformerModel(){
            return loadLocalTransformerModel();
        }

        /**
         * Legacy compatibility path for callers that still need the old concrete executor plus old lifecycle behavior.
         */
        @Deprecated
        public AbstractModel buildAbstractModel(){
            AbstractModel model = loadLocalTransformerModel();
            if (parallelSettings.isPresent()) {
                GossipParallelMembership membership = GossipParallelMembership.start(parallelSettings.get());
                model.setGossipParallelMembership(membership);
                membership.startWorkerWhenReady(this);
            }
            return model;
        }

        private AbstractModel loadLocalTransformerModel(){
            System.setProperty("jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK", this.oobCheck);
            ModelFetcher fetcherForLoad = resolveModelFetcherForLoad();
            File modelRoot = fetcherForLoad.maybeDownload();
            if (pool == null){
                pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
            }
            if (provider == null){
                ConfigurableTensorProvider base = new ConfigurableTensorProvider(allocator, pool);
                Optional<TensorOperations> maybe = getNative(base.get());
                provider = maybe.map(ConfigurableTensorProvider::new).orElse(base);
            }
            AbstractModel model = ModelSupport.loadModel(modelRoot, workingMem, workingQuant, provider,
                    mr, allocator, settings, fetcherForLoad, toolCallParser, pool, tensorParallelContext, tensorParallelCollectives,
                    outputHeadQuantization);
            return model;
        }

        ModelFetcher resolveModelFetcherForLoad() {
            fetch.setDownload(download);
            if (quantizeOnDemand.isEmpty()) {
                return fetch;
            }
            QuantizeOnDemand quantize = quantizeOnDemand.get();
            ModelFetcher target = cachePeerFetcher(quantize.outputOwner(), quantize.outputModel()).withDownload(false);
            if (isLocallyComplete(target)) {
                LOGGER.info("Using existing quantized model target {}", target.pathForModel());
                return target;
            }
            if (Files.exists(target.pathForModel())) {
                throw new IllegalStateException("Quantized target exists but is incomplete: " + target.pathForModel());
            }

            LOGGER.info("Quantized model target {} is missing; resolving source {}", target.pathForModel(), fetch.pathForModel());
            Path sourceDir = fetch.maybeDownload().toPath();
            Path targetDir = target.pathForModel();
            Path stagingDir = targetDir.resolveSibling(targetDir.getFileName() + ".tmp-" + UUID.randomUUID());
            try {
                Files.createDirectories(targetDir.getParent());
                LOGGER.info("Creating quantized model target {} via staging directory {}", targetDir, stagingDir);
                new ModelQuantizer().quantizeModelDirectory(sourceDir, stagingDir, quantize.targetType(),
                        ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER);
                Files.move(stagingDir, targetDir);
                LOGGER.info("Installed quantized model target {}", targetDir);
            } catch (IOException e) {
                deleteQuietly(stagingDir);
                throw new RuntimeException("Unable to install quantized model at " + targetDir, e);
            } catch (RuntimeException e) {
                deleteQuietly(stagingDir);
                throw e;
            }
            return target;
        }

        private boolean isLocallyComplete(ModelFetcher fetcher) {
            try {
                fetcher.maybeDownload();
                return true;
            } catch (IllegalStateException e) {
                return false;
            }
        }

        private ModelFetcher cachePeerFetcher(String owner, String name) {
            ModelFetcher peer = new ModelFetcher(owner, name);
            peer.setBaseDir(fetch.getBaseDir());
            return peer;
        }

        private void deleteQuietly(Path directory) {
            if (directory == null || !Files.exists(directory)) {
                return;
            }
            try (var paths = Files.walk(directory)) {
                paths.sorted(Comparator.reverseOrder())
                        .forEach(path -> {
                            try {
                                Files.deleteIfExists(path);
                            } catch (IOException ignored) {
                                LOGGER.warn("unable to delete staging path {}", path);
                            }
                        });
            } catch (IOException e) {
                LOGGER.warn("unable to clean quantized-model staging directory {}", directory, e);
            }
        }

        public ModelFetcher getFetch() {
            return fetch;
        }

        public MetricRegistry getMr() {
            return mr;
        }

        public TensorAllocator getAllocator() {
            return allocator;
        }

        public DType getWorkingMem() {
            return workingMem;
        }

        public DType getWorkingQuant() {
            return workingQuant;
        }

        public KvBufferCacheSettings getSettings() {
            return settings;
        }

        public ConfigurableTensorProvider getProvider() {
            return provider;
        }

        public WrappedForkJoinPool getPool() {
            return this.pool;
        }

        public TensorParallelContext getTensorParallelContext() {
            return tensorParallelContext;
        }

        public TensorParallelCollectives getTensorParallelCollectives() {
            return tensorParallelCollectives;
        }

        public Optional<GossipParallelSettings> getParallelSettings() {
            return parallelSettings;
        }
    }

    public static Optional<TensorOperations> getNative(TensorOperations inject){
        String nm = "io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations";
        try {
            return Optional.of((TensorOperations) Class.forName(nm)
                    .getConstructor(TensorOperations.class).newInstance(inject));
        } catch (InstantiationException | ClassNotFoundException | NoSuchMethodException |
                 InvocationTargetException | IllegalAccessException e) {
            LOGGER.warn("unable to load native SIMD support", e);
        } catch (UnsatisfiedLinkError e){
            LOGGER.warn("unable to load native SIMD support", e);
        }
        return Optional.empty();
    }
}
