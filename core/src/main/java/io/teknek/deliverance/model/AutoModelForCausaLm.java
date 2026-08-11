package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.gemma2.Gemma2Config;
import io.teknek.deliverance.model.gemma2.Gemma2Model;
import io.teknek.deliverance.model.gemma3.Gemma3Config;
import io.teknek.deliverance.model.gemma3.Gemma3Model;
import io.teknek.deliverance.model.gemma4.Gemma4Config;
import io.teknek.deliverance.model.gemma4.Gemma4Model;
import io.teknek.deliverance.model.gpt2.Gpt2Config;
import io.teknek.deliverance.model.gpt2.Gpt2Model;
import io.teknek.deliverance.model.granitemoehybrid.GraniteMoeHybridConfig;
import io.teknek.deliverance.model.granitemoehybrid.GraniteMoeHybridModel;
import io.teknek.deliverance.model.llama.LlamaConfig;
import io.teknek.deliverance.model.llama.LlamaModel;
import io.teknek.deliverance.model.mistral.MistralConfig;
import io.teknek.deliverance.model.mistral.MistralModel;
import io.teknek.deliverance.model.mixtral.MixtralConfig;
import io.teknek.deliverance.model.mixtral.MixtralModel;
import io.teknek.deliverance.model.qwen2.Qwen2Config;
import io.teknek.deliverance.model.qwen2.Qwen2Model;
import io.teknek.deliverance.model.qwen3.Qwen3Config;
import io.teknek.deliverance.model.qwen3.Qwen3Model;
import io.teknek.deliverance.model.qwen3.Qwen3MoeConfig;
import io.teknek.deliverance.model.qwen3.Qwen3MoeModel;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.SingleRankTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.GossipParallelMembership;
import io.teknek.deliverance.model.tensorparallel.GossipParallelSettings;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.LoraAdapter;
import io.teknek.deliverance.safetensors.MergingWeightLoader;
import io.teknek.deliverance.safetensors.ModelQuantizer;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.safetensors.fetch.LoraAdapterModelFetcher;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.MachineSpec;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import io.teknek.deliverance.tensor.operations.PanamaTensorOperations;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.tensorlib.TensorRuntimeMode;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import io.teknek.deliverance.toolcallparser.LlamaToolCallParser;
import io.teknek.deliverance.toolcallparser.QwenToolCallParser;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import io.teknek.sketches.SketchesSettings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.lang.reflect.InvocationTargetException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
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
        if (fetcher.getName().startsWith("antares") || fetcher.getName().startsWith("granite-4.0-h")) {
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
        private boolean tensorProviderExplicit;
        private WrappedForkJoinPool pool;
        private String oobCheck = "2";
        private TensorParallelContext tensorParallelContext = new StaticTensorParallelContext(0, 1);
        private TensorParallelCollectives tensorParallelCollectives = new SingleRankTensorParallelCollectives();
        private Optional<GossipParallelSettings> parallelSettings = Optional.empty();
        private Optional<DType> outputHeadQuantization = Optional.empty();
        private Optional<QuantizeOnDemand> quantizeOnDemand = Optional.empty();
        private LoraAdapterModelFetcher loraAdapterFetcher;
        private final EnumMap<TensorProviderKind, TensorOperations> additionalTensorOperations = new EnumMap<>(TensorProviderKind.class);
        private boolean download = true;
        private int maxBatchSize = AbstractModel.DEFAULT_MAX_BATCH_SIZE;
        private SketchesSettings sketchesSettings = SketchesSettings.DEFAULT;
        private boolean gpuPrefill;
        private boolean gpuDecode;
        private boolean gpuDecodeAttention;
        private Optional<TensorRuntimeMode> tensorRuntimeMode = Optional.empty();

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
        public Builder withSketchesSettings(SketchesSettings sketchesSettings){
            this.sketchesSettings = Objects.requireNonNull(sketchesSettings, "sketchesSettings");
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
            this.tensorProviderExplicit = true;
            return this;
        }
        public Builder withAdditionalTensorOperations(TensorProviderKind kind, TensorOperations operations) {
            this.additionalTensorOperations.put(Objects.requireNonNull(kind, "kind"),
                    Objects.requireNonNull(operations, "operations"));
            return this;
        }
        public Builder withoutAdditionalTensorOperations(TensorProviderKind kind) {
            this.additionalTensorOperations.remove(Objects.requireNonNull(kind, "kind"));
            return this;
        }
        public Builder withToolCallParser(ToolCallParser toolCallParser){
            this.toolCallParser = toolCallParser;
            return this;
        }

        ToolCallParser toolCallParserForTest() {
            return toolCallParser;
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

        public Builder withMaxBatchSize(int maxBatchSize) {
            if (maxBatchSize < 1) {
                throw new IllegalArgumentException("maxBatchSize must be >= 1");
            }
            this.maxBatchSize = maxBatchSize;
            return this;
        }

        /**
         * Opts into experimental GPU prefill work. Current behavior loads GPU tensor operations and makes them available
         * through {@link AbstractModel#tensorOperations(TensorProviderKind)} for targeted prefill matmul experiments.
         */
        public Builder withGpuPrefill(boolean gpuPrefill) {
            this.gpuPrefill = gpuPrefill;
            return this;
        }

        /**
         * Opts into experimental GPU decode projection work. Decode runs once per generated token, so routing supported
         * projection matmuls to GPU is controlled separately from prefill.
         */
        public Builder withGpuDecode(boolean gpuDecode) {
            this.gpuDecode = gpuDecode;
            return this;
        }

        public Builder withGpuDecodeAttention(boolean gpuDecodeAttention) {
            this.gpuDecodeAttention = gpuDecodeAttention;
            return this;
        }

        public Builder withTensorRuntimeMode(TensorRuntimeMode tensorRuntimeMode) {
            this.tensorRuntimeMode = Optional.of(Objects.requireNonNull(tensorRuntimeMode, "tensorRuntimeMode"));
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

        /**
         * Merges the given LoRA/PEFT adapter into the base model's weights at load time (Phase 1
         * "merge-at-load" -- see {@code
         * StepPlans/deliverance_lora_step3_merging_weightloader_plan_v1.md}). The base model is
         * loaded and merged fresh on every {@link #build()}/{@link #buildLocalTransformerModel()}
         * call; nothing is cached to disk. Requires a dense (F32/BF16/F16) base model and does not
         * support tensor-parallel loading -- both are enforced with a clear exception at load time
         * rather than silently producing an unmerged or wrong model.
         */
        public Builder withLoraAdapter(LoraAdapterModelFetcher adapterFetcher) {
            this.loraAdapterFetcher = adapterFetcher;
            return this;
        }

        /** Applies a JSON-friendly builder configuration object. Explicit method calls made after this one can override it. */
        public Builder withConfig(AutoModelConfig config) {
            Objects.requireNonNull(config, "config");
            config.workingMemoryType().ifPresent(this::withWorkingMemoryType);
            config.workingQuantType().ifPresent(this::withWorkingQuantType);
            config.outputHeadQuantization().ifPresent(this::withOutputHeadQuantization);
            config.gpuPrefill().ifPresent(this::withGpuPrefill);
            config.gpuDecode().ifPresent(this::withGpuDecode);
            config.gpuDecodeAttention().ifPresent(this::withGpuDecodeAttention);
            config.download().ifPresent(this::withDownload);
            config.maxBatchSize().ifPresent(this::withMaxBatchSize);
            config.tensorRuntimeMode().ifPresent(this::withTensorRuntimeMode);
            config.kvBufferCache().map(AutoModelConfig.KvBufferCache::toSettings).ifPresent(this::withKvBufferCacheSettings);
            config.quantizeOnDemand().ifPresent(q -> withQuantizeOnDemand(q.targetType(), q.outputOwner(), q.outputModel()));
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
            copy.tensorProviderExplicit = this.tensorProviderExplicit;
            copy.pool = this.pool;
            copy.oobCheck = this.oobCheck;
            copy.tensorParallelContext = context;
            copy.tensorParallelCollectives = Objects.requireNonNull(collectives, "collectives");
            copy.outputHeadQuantization = this.outputHeadQuantization;
            copy.quantizeOnDemand = this.quantizeOnDemand;
            copy.loraAdapterFetcher = this.loraAdapterFetcher;
            copy.additionalTensorOperations.putAll(this.additionalTensorOperations);
            copy.download = this.download;
            copy.maxBatchSize = this.maxBatchSize;
            copy.sketchesSettings = this.sketchesSettings;
            copy.gpuPrefill = this.gpuPrefill;
            copy.gpuDecode = this.gpuDecode;
            copy.gpuDecodeAttention = this.gpuDecodeAttention;
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
            return DefaultCausalLanguageModel.local(model, sketchesSettings);
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
            Optional<LoraAdapter> loraAdapter = loraAdapterFetcher == null
                    ? Optional.empty()
                    : Optional.of(LoraAdapter.fromPretrained(loraAdapterFetcher, mr));
            AbstractModel model = constructModel(modelRoot, loraAdapter);
            model.setMaxBatchSize(maxBatchSize);
            model.setTensorProviderExplicit(tensorProviderExplicit);
            model.setGpuPrefillEnabled(gpuPrefill);
            model.setGpuDecodeEnabled(gpuDecode);
            model.setGpuDecodeAttentionEnabled(gpuDecodeAttention);
            model.setTensorRuntimeMode(tensorRuntimeMode);
            if (!tensorProviderExplicit) {
                model.addTensorOperations(hydrateTensorOperations());
            }
            model.init();
            return model;
        }

        protected AbstractModel constructModel(File modelRoot, Optional<LoraAdapter> loraAdapter) {
            File configFile = modelRoot.toPath().resolve("config.json").toFile();
            if (!configFile.exists()) {
                throw new RuntimeException("Expecting to find config file " + configFile);
            }
            try {
                String modelType = JsonUtils.om.readTree(configFile).get("model_type").textValue().toUpperCase();
                Config config = readConfig(configFile, modelType);
                PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(modelRoot.toPath());
                WeightLoader weightLoader = new DefaultWeightLoader(modelRoot);
                if (loraAdapter.isPresent()) {
                    weightLoader = new MergingWeightLoader(weightLoader, loraAdapter.get(), provider.get());
                }
                return newModel(modelType, config, weightLoader, tokenizer);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        public Config readConfig(File configFile, String modelType) throws IOException {
            return switch (modelType) {
                case "BERT" -> JsonUtils.om.readValue(configFile, io.teknek.deliverance.model.bert.BertConfig.class);
                case "LLAMA" -> JsonUtils.om.readValue(configFile, LlamaConfig.class);
                case "QWEN2" -> JsonUtils.om.readValue(configFile, Qwen2Config.class);
                case "QWEN3" -> JsonUtils.om.readValue(configFile, Qwen3Config.class);
                case "QWEN3_MOE" -> JsonUtils.om.readValue(configFile, Qwen3MoeConfig.class);
                case "GEMMA2" -> JsonUtils.om.readValue(configFile, Gemma2Config.class);
                case "GEMMA4" -> JsonUtils.om.readValue(configFile, Gemma4Config.class);
                case "GEMMA3_TEXT" -> JsonUtils.om.readValue(configFile, Gemma3Config.class);
                case "MISTRAL" -> JsonUtils.om.readValue(configFile, MistralConfig.class);
                case "GPT2" -> JsonUtils.om.readValue(configFile, Gpt2Config.class);
                case "MIXTRAL" -> JsonUtils.om.readValue(configFile, MixtralConfig.class);
                case "GRANITEMOEHYBRID" -> JsonUtils.om.readValue(configFile, GraniteMoeHybridConfig.class);
                default -> throw new IllegalArgumentException(modelType + " not found in AutoModelForCausaLm");
            };
        }

        protected AbstractModel newModel(String modelType, Config config, WeightLoader weightLoader,
                PreTrainedTokenizer tokenizer) {
            return switch (modelType) {
                case "LLAMA" -> new LlamaModel(AbstractModel.InferenceType.FULL_GENERATION, config, weightLoader,
                        tokenizer, workingMem, workingQuant, Optional.empty(), provider, mr, allocator, settings,
                        toolCallParser, pool, tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
                case "QWEN2" -> new Qwen2Model(AbstractModel.InferenceType.FULL_GENERATION, config, weightLoader,
                        tokenizer, workingMem, workingQuant, Optional.empty(), provider, mr, allocator, settings,
                        toolCallParser, pool, tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
                case "QWEN3" -> new Qwen3Model(AbstractModel.InferenceType.FULL_GENERATION, config, weightLoader,
                        tokenizer, workingMem, workingQuant, Optional.empty(), provider, mr, allocator, settings,
                        toolCallParser, pool, tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
                case "QWEN3_MOE" -> new Qwen3MoeModel(AbstractModel.InferenceType.FULL_GENERATION, config, weightLoader,
                        tokenizer, workingMem, workingQuant, Optional.empty(), provider, mr, allocator, settings,
                        toolCallParser, pool, tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
                case "GEMMA2" -> new Gemma2Model(AbstractModel.InferenceType.FULL_GENERATION, config, weightLoader,
                        tokenizer, workingMem, workingQuant, Optional.empty(), provider, mr, allocator, settings,
                        toolCallParser, pool, tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
                case "GEMMA4" -> new Gemma4Model(AbstractModel.InferenceType.FULL_GENERATION, config, weightLoader,
                        tokenizer, workingMem, workingQuant, Optional.empty(), provider, mr, allocator, settings,
                        toolCallParser, pool, tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
                case "GEMMA3_TEXT" -> new Gemma3Model(AbstractModel.InferenceType.FULL_GENERATION, config, weightLoader,
                        tokenizer, workingMem, workingQuant, Optional.empty(), provider, mr, allocator, settings,
                        toolCallParser, pool, tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
                case "MISTRAL" -> new MistralModel(AbstractModel.InferenceType.FULL_GENERATION, config, weightLoader,
                        tokenizer, workingMem, workingQuant, Optional.empty(), provider, mr, allocator, settings,
                        toolCallParser, pool, tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
                case "GPT2" -> new Gpt2Model(AbstractModel.InferenceType.FULL_GENERATION, config, weightLoader,
                        tokenizer, workingMem, workingQuant, Optional.empty(), provider, mr, allocator, settings,
                        toolCallParser, pool, tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
                case "MIXTRAL" -> new MixtralModel(AbstractModel.InferenceType.FULL_GENERATION, config, weightLoader,
                        tokenizer, workingMem, workingQuant, Optional.empty(), provider, mr, allocator, settings,
                        toolCallParser, pool, tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
                case "GRANITEMOEHYBRID" -> new GraniteMoeHybridModel(AbstractModel.InferenceType.FULL_GENERATION,
                        config, weightLoader, tokenizer, workingMem, workingQuant, Optional.empty(), provider, mr,
                        allocator, settings, toolCallParser, pool, tensorParallelContext, tensorParallelCollectives,
                        outputHeadQuantization);
                default -> throw new IllegalArgumentException(modelType + " not supported by AutoModelForCausaLm");
            };
        }

        private Map<TensorProviderKind, TensorOperations> hydrateTensorOperations() {
            EnumMap<TensorProviderKind, TensorOperations> operations = new EnumMap<>(TensorProviderKind.class);

            TensorOperations naive = additionalTensorOperations.getOrDefault(TensorProviderKind.NAIVE, new NaiveTensorOperations());
            operations.put(TensorProviderKind.NAIVE, naive);

            TensorOperations panama = additionalTensorOperations.getOrDefault(TensorProviderKind.PANAMA,
                    new PanamaTensorOperations(MachineSpec.VECTOR_TYPE, allocator, pool));
            operations.put(TensorProviderKind.PANAMA, panama);

            TensorOperations simd = additionalTensorOperations.get(TensorProviderKind.SIMD);
            if (simd == null) {
                simd = getNative(panama).orElse(panama);
            }
            operations.put(TensorProviderKind.SIMD, simd);

            TensorOperations gpu = additionalTensorOperations.get(TensorProviderKind.GPU);
            if (gpu == null) {
                Optional<TensorOperations> maybeGpu = tryLoadTensorOperations("io.teknek.deliverance.tensor.operations.NativeGPUTensorOperations");
                maybeGpu.ifPresent(value -> operations.put(TensorProviderKind.GPU, value));
                if ((gpuPrefill || gpuDecode || gpuDecodeAttention) && maybeGpu.isEmpty()) {
                    throw new IllegalStateException("GPU projection requested but NativeGPUTensorOperations is not available");
                }
            } else {
                operations.put(TensorProviderKind.GPU, gpu);
            }

            return Map.copyOf(operations);
        }

        private Optional<TensorOperations> tryLoadTensorOperations(String className) {
            try {
                return Optional.of((TensorOperations) Class.forName(className).getConstructor().newInstance());
            } catch (Throwable t) {
                LOGGER.debug("tensor operations provider {} is not available", className, t);
                return Optional.empty();
            }
        }

        public ModelFetcher resolveModelFetcherForLoad() {
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

        public Optional<DType> getOutputHeadQuantization() {
            return outputHeadQuantization;
        }

        public boolean isDownload() {
            return download;
        }

        public int getMaxBatchSize() {
            return maxBatchSize;
        }

        public boolean isGpuPrefill() {
            return gpuPrefill;
        }

        public boolean isGpuDecode() {
            return gpuDecode;
        }

        public boolean isGpuDecodeAttention() {
            return gpuDecodeAttention;
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
