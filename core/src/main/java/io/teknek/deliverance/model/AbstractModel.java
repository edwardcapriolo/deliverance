package io.teknek.deliverance.model;

import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;

import java.nio.FloatBuffer;
import java.util.*;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.stream.Collectors;


import io.teknek.deliverance.CausualWhisperer;
import io.teknek.deliverance.Classifier;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.classifier.ClassifyOutput;
import io.teknek.deliverance.embedding.PoolingLayer;

import io.teknek.deliverance.embedding.PoolingType;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.guided.LogitsProcessor;
import io.teknek.deliverance.grace.EncodeOptions;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.math.VectorMathUtils;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.SingleRankTensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.GossipParallelMembership;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.TensorParallelPlanner;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.LoraAdapter;
import io.teknek.deliverance.safetensors.LoraLayerDelta;
import io.teknek.deliverance.safetensors.ResolvedLoraAdapter;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.safetensors.fetch.LoraAdapterModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.*;
import io.teknek.deliverance.tensor.kv.KvCacheManager;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.TensorOperations;
import io.teknek.deliverance.tensorlib.PlannedTensor;
import io.teknek.deliverance.tensorlib.TensorPlan;
import io.teknek.deliverance.tensorlib.TensorRuntime;
import io.teknek.deliverance.tensorlib.TensorRuntimeMode;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import jdk.incubator.vector.FloatVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static io.teknek.deliverance.tensor.DebugSupport.debug;

/**
 * Base implementation for generation, classification, embedding, and shared model utilities.
 *
 * <h2>Prefix KV-cache contract</h2>
 * <p>Generation can reuse block-aligned KV prefixes through {@link KvBufferCache}. The cache is an internal
 * performance path: it avoids recomputing already-seen prompt prefixes, then runs any uncached suffix tokens and
 * begins decoding after the full prompt length. The position invariant is strict: cache hits must not change the
 * decode start position or the token budget. For example, with an 8-token cached prefix and a 9-token prompt, the
 * first generated token belongs at position 9, not 17.</p>
 *
 * <h2>What this does not guarantee</h2>
 * <p>This class does not guarantee that generated text is exactly identical between a cold full-prefill request and
 * a cache-hit request. That stronger property requires batch/chunk-invariant kernels. In practice, full prefill and
 * split prefill can differ numerically because matrix multiplication, attention, RMSNorm, and activation
 * quantization may use different reduction strategies or scaling decisions for different batch/chunk shapes. This is
 * consistent with the behavior of common inference engines unless they explicitly enable deterministic,
 * batch-invariant kernels.</p>
 *
 * <p>Useful background: Thinking Machines, "Defeating Nondeterminism in LLM Inference",
 * https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/</p>
 */
public abstract class AbstractModel implements Generator, Classifier {
    static final Logger logger = LoggerFactory.getLogger(AbstractModel.class);

    public static final int DEFAULT_MAX_BATCH_SIZE = 512;
    private static final long PREFILL_PROGRESS_INTERVAL_NANOS = TimeUnit.SECONDS.toNanos(3);
    private static final ThreadLocal<PrefillProgress> PREFILL_PROGRESS = new ThreadLocal<>();

    private static final class PrefillProgress {
        private final int totalTokens;
        private final int startPos;
        private final long startNanos;
        private int chunkStart;
        private int chunkTokens;
        private long nextLogNanos;

        private PrefillProgress(int totalTokens, int startPos, long startNanos) {
            this.totalTokens = totalTokens;
            this.startPos = startPos;
            this.startNanos = startNanos;
            this.nextLogNanos = startNanos + PREFILL_PROGRESS_INTERVAL_NANOS;
        }
    }

    public enum GenerationDebugEventType {
        AFTER_PREFIX_COPY,
        AFTER_PROMPT_PREFILL
    }

    public record GenerationDebugEvent(
            GenerationDebugEventType type,
            int[] promptTokens,
            int prefixLength,
            int startPos,
            int tokensToProcessLength,
            KvBufferCache.KvBuffer kvBuffer
    ) {
    }

    public record LayerDebugEvent(
            int layerIndex,
            String stage,
            TensorParallelContext tensorParallelContext,
            AbstractTensor hiddenStates
    ) {
    }

    /**
     * Forward execution boundary used by generation coordinators that do not own local KV memory directly.
     */
    public interface GenerationForwarder {
        AbstractTensor batchForward(int[] tokenIds, int startPosition);

        AbstractTensor forward(int tokenId, int position);
    }

    public enum InferenceType {
        // Used for distributed inference
        INPUT_TO_EMBEDDING(true, false, false, false, false),
        OUTPUT_TO_TOKEN(false, false, true, false, false),
        FORWARD_PASS(true, true, false, false, false),

        // Used for different types of inference
        FULL_GENERATION(true, true, true, false, false),
        FULL_CLASSIFICATION(true, true, false, true, true),
        FULL_EMBEDDING(true, true, false, false, true);

        final boolean isInput;
        final boolean isOutput;
        final boolean isClassify;
        final boolean isFwdPass;
        final boolean isPooling;

        InferenceType(boolean isInput, boolean isFwdPass, boolean isOutput, boolean isClassify, boolean isPooling) {
            this.isInput = isInput;
            this.isOutput = isOutput;
            this.isFwdPass = isFwdPass;
            this.isClassify = isClassify;
            this.isPooling = isPooling;
        }
    }

    protected final InferenceType inferenceType;
    protected final Config config;
    protected final WeightLoader weights;
    protected final PreTrainedTokenizer tokenizer;
    protected final DType modelDType;
    protected final DType workingDType;
    protected final DType workingQType;
    protected final Optional<DType> modelQType;
    protected final Optional<DType> outputHeadQuantization;
    protected EmbedInput embedInput;
    protected SampleOutput sampleOutput;
    protected TransformerBlock[] transformerBlocks;
    protected KvBufferCache kvBufferCache;
    protected KvCacheManager kvCacheManager;
    protected final ConfigurableTensorProvider configurableTensorProvider;
    protected final MetricRegistry metricRegistry;
    protected final TensorAllocator tensorAllocator;
    protected final TensorParallelContext tensorParallelContext;
    protected final TensorParallelCollectives tensorParallelCollectives;
    private final EnumMap<TensorProviderKind, TensorOperations> tensorOperations = new EnumMap<>(TensorProviderKind.class);
    private boolean tensorProviderExplicit;
    private boolean gpuPrefillEnabled;
    private boolean gpuDecodeEnabled;
    private boolean gpuDecodeAttentionEnabled;
    private boolean gpuDiffusionBlockProjectionEnabled;
    private boolean packedBlockAttentionEnabled;
    private boolean packedPrefillEnabled = true;
    private GossipParallelMembership gossipParallelMembership;

    //embedding
    protected Optional<PoolingLayer> poolingLayer;

    protected final ToolCallParser toolCallParser;

    protected ClassifyOutput classifyOutput;
    protected WrappedForkJoinPool pool;
    protected PreTrainedTokenizer preTrainedTokenizer;
    protected int maxBatchSize = DEFAULT_MAX_BATCH_SIZE;
    protected TensorPlan modelLineagePlan;
    private final ConcurrentMap<String, TensorPlan.ImmutableTensor> modelLineageTensors = new ConcurrentHashMap<>();
    private final Queue<ModelLineageEntry> modelLineageEntries = new ConcurrentLinkedQueue<>();
    private Optional<TensorRuntimeMode> tensorRuntimeMode = Optional.empty();
    private TensorRuntime tensorRuntime;
    private Map<String, Object> generationOptions = Map.of();
    private boolean initialized;
    private boolean tensorPlanTraceEnabled;
    private final ConcurrentMap<UUID, TensorPlanTraceContext> tensorPlanTraces = new ConcurrentHashMap<>();
    private final ThreadLocal<UUID> activeTensorPlanTraceId = new ThreadLocal<>();
    private volatile Consumer<GenerationDebugEvent> generationDebugHook = event -> {};
    private volatile Consumer<LayerDebugEvent> layerDebugHook = event -> {};

    protected AbstractModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer t, DType workingMemoryDType,
                            DType workingMemoryQType, Optional<DType> modelQType, ConfigurableTensorProvider provider,
                            MetricRegistry metricRegistry, TensorAllocator tensorAllocator, KvBufferCacheSettings kvBufferCacheSettings,
                            ToolCallParser toolCallParser, WrappedForkJoinPool pool) {
        this(inferenceType, c, w, t, workingMemoryDType, workingMemoryQType, modelQType, provider, metricRegistry,
                tensorAllocator, kvBufferCacheSettings, toolCallParser, pool, new StaticTensorParallelContext(0, 1),
                new SingleRankTensorParallelCollectives());
    }

    protected AbstractModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer t, DType workingMemoryDType,
                            DType workingMemoryQType, Optional<DType> modelQType, ConfigurableTensorProvider provider,
                            MetricRegistry metricRegistry, TensorAllocator tensorAllocator, KvBufferCacheSettings kvBufferCacheSettings,
                            ToolCallParser toolCallParser, WrappedForkJoinPool pool,
                            TensorParallelContext tensorParallelContext) {
        this(inferenceType, c, w, t, workingMemoryDType, workingMemoryQType, modelQType, provider, metricRegistry,
                tensorAllocator, kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext,
                new SingleRankTensorParallelCollectives());
    }

    protected AbstractModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer t, DType workingMemoryDType,
                            DType workingMemoryQType, Optional<DType> modelQType, ConfigurableTensorProvider provider,
                            MetricRegistry metricRegistry, TensorAllocator tensorAllocator, KvBufferCacheSettings kvBufferCacheSettings,
                            ToolCallParser toolCallParser, WrappedForkJoinPool pool,
                            TensorParallelContext tensorParallelContext,
                            TensorParallelCollectives tensorParallelCollectives) {
        this(inferenceType, c, w, t, workingMemoryDType, workingMemoryQType, modelQType, provider, metricRegistry,
                tensorAllocator, kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext,
                tensorParallelCollectives, Optional.empty());
    }

    protected AbstractModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer t, DType workingMemoryDType,
                            DType workingMemoryQType, Optional<DType> modelQType, ConfigurableTensorProvider provider,
                            MetricRegistry metricRegistry, TensorAllocator tensorAllocator, KvBufferCacheSettings kvBufferCacheSettings,
                            ToolCallParser toolCallParser, WrappedForkJoinPool pool,
                            TensorParallelContext tensorParallelContext,
                            TensorParallelCollectives tensorParallelCollectives,
                            Optional<DType> outputHeadQuantization) {
        this.inferenceType = inferenceType;
        this.config = c;
        this.weights = w;
        this.tokenizer = t;
        this.tensorParallelContext = Objects.requireNonNull(tensorParallelContext, "tensorParallelContext");
        this.tensorParallelCollectives = Objects.requireNonNull(tensorParallelCollectives, "tensorParallelCollectives");
        this.outputHeadQuantization = Objects.requireNonNull(outputHeadQuantization, "outputHeadQuantization");
        TensorParallelPlanner.validate(c, tensorParallelContext);

        this.modelDType = w.getModelDType();
        this.workingDType = workingMemoryDType;
        this.modelQType = modelQType;
        this.configurableTensorProvider = provider;
        this.tensorOperations.put(TensorProviderKind.SIMD, provider.get());
        this.metricRegistry = metricRegistry;
        this.tensorAllocator = tensorAllocator;
        this.kvBufferCache = new KvBufferCache(this, kvBufferCacheSettings);
        this.kvCacheManager = new KvCacheManager(c.numberOfLayers, c.contextLength,
                c.kvLength / tensorParallelContext.size(), workingMemoryDType, kvBufferCacheSettings, tensorAllocator,
                metricRegistry);
        this.toolCallParser = toolCallParser;

        this.workingQType = resolveWorkingQType(workingMemoryQType);

        this.pool = pool;
        logger.debug("model constructor complete config={} inference_type={}", config.getClass().getSimpleName(), inferenceType);
    }

    public final void init() {
        if (initialized) {
            return;
        }
        this.modelLineagePlan = new TensorPlan(configurableTensorProvider.get(), pool, metricRegistry, this, tensorRuntime);
        logger.debug("model init start config={} inference_type={}", config.getClass().getSimpleName(), inferenceType);
        this.embedInput = inferenceType.isInput ? loadInputWeights() : null;
        this.transformerBlocks = inferenceType.isFwdPass ? loadTransformerBlockWeights() : null;
        this.sampleOutput = inferenceType.isOutput ? loadOutputWeights() : null;
        this.classifyOutput = inferenceType.isClassify ? loadClassifierWeights() : null;
        this.poolingLayer = inferenceType.isPooling ? Optional.ofNullable(loadPoolingWeights()) : Optional.empty();
        initialized = true;
        logger.debug("model init complete config={} inference_type={} layers={}", config.getClass().getSimpleName(),
                inferenceType, this.transformerBlocks == null ? 0 : this.transformerBlocks.length);
    }

    private DType resolveWorkingQType(DType requestedWorkingQType) {
        if (requestedWorkingQType == null) {
            requestedWorkingQType = configurableTensorProvider.get().preferredWorkingQuantizedType();
        }

        // FIXME: This is a hack to support Avoid Q8F32 evals
        if (modelDType == DType.F32 && requestedWorkingQType != DType.F32 && modelQType.isEmpty()) {
            requestedWorkingQType = DType.F32;
        }

        // FIXME: This is a hack to support Avoid Q8BF16 evals
        if (modelDType == DType.BF16 && requestedWorkingQType != DType.BF16 && requestedWorkingQType != DType.F32
                && modelQType.isEmpty()) {
            requestedWorkingQType = DType.BF16;
        }

        // Check to make sure the model is big enough to support Q4I8 computations.
        if (modelDType == DType.Q4
                && requestedWorkingQType == DType.I8
                && ((config.embeddingLength / Q8ByteBufferTensor.BLOCK_SIZE) % (FloatVector.SPECIES_PREFERRED.vectorBitSize() / Float.SIZE) != 0
                || (config.hiddenLength / Q8ByteBufferTensor.BLOCK_SIZE) % (FloatVector.SPECIES_PREFERRED.vectorBitSize() / Float.SIZE) != 0)) {
            requestedWorkingQType = DType.F32;
        }

        // Check to make sure the model is big enough to support Q4I8 computations.
        if (modelDType == DType.Q4
                && requestedWorkingQType == DType.I8
                && (config.embeddingLength / Q8ByteBufferTensor.BLOCK_SIZE) % (FloatVector.SPECIES_PREFERRED.vectorBitSize() / Float.SIZE) != 0) {
            logger.warn("Determined model could not support quant type. Request {} model {} falling back to {} ",
                    requestedWorkingQType, modelDType, DType.F32);
            requestedWorkingQType = DType.F32;
        }

        // Some operation providers don't support Q4I8.
        DType providerPreferredQType = configurableTensorProvider.get().preferredWorkingQuantizedType();
        if (modelDType == DType.Q4 && requestedWorkingQType.size() < providerPreferredQType.size()) {
            requestedWorkingQType = providerPreferredQType;
            logger.warn("Tensor provider {} does not support Q4. Using {} as workingMemoryType ",
                    configurableTensorProvider.get().name(), requestedWorkingQType);
        }

        if (requestedWorkingQType == workingDType) {
            return requestedWorkingQType;
        }
        try (AbstractTensor tmp = makeDenseTensor(Q8ByteBufferTensor.BLOCK_SIZE);
             AbstractTensor tmp2 = configurableTensorProvider.get().quantize(tmp, requestedWorkingQType, 0,
                     Q8ByteBufferTensor.BLOCK_SIZE)) {
            if (tmp2.dType() == requestedWorkingQType) {
                return requestedWorkingQType;
            }
            logger.warn("Quantized memory type {} not supported, falling back to {}", requestedWorkingQType,
                    workingDType);
            return workingDType;
        }
    }

    void addTensorOperations(Map<TensorProviderKind, TensorOperations> additionalTensorOperations) {
        this.tensorOperations.putAll(additionalTensorOperations);
    }

    void setTensorProviderExplicit(boolean tensorProviderExplicit) {
        this.tensorProviderExplicit = tensorProviderExplicit;
    }

    void setGpuPrefillEnabled(boolean gpuPrefillEnabled) {
        this.gpuPrefillEnabled = gpuPrefillEnabled;
    }

    void setGpuDecodeEnabled(boolean gpuDecodeEnabled) {
        this.gpuDecodeEnabled = gpuDecodeEnabled;
    }

    void setGpuDecodeAttentionEnabled(boolean gpuDecodeAttentionEnabled) {
        this.gpuDecodeAttentionEnabled = gpuDecodeAttentionEnabled;
    }

    void setGpuDiffusionBlockProjectionEnabled(boolean gpuDiffusionBlockProjectionEnabled) {
        this.gpuDiffusionBlockProjectionEnabled = gpuDiffusionBlockProjectionEnabled;
    }

    void setPackedBlockAttentionEnabled(boolean packedBlockAttentionEnabled) {
        this.packedBlockAttentionEnabled = packedBlockAttentionEnabled;
    }

    void setPackedPrefillEnabled(boolean packedPrefillEnabled) {
        this.packedPrefillEnabled = packedPrefillEnabled;
    }

    public boolean isGpuPrefillEnabled() {
        return gpuPrefillEnabled;
    }

    public boolean isGpuDecodeEnabled() {
        return gpuDecodeEnabled;
    }

    public boolean isGpuDecodeAttentionEnabled() {
        return gpuDecodeAttentionEnabled;
    }

    public boolean isGpuDiffusionBlockProjectionEnabled() {
        return gpuDiffusionBlockProjectionEnabled;
    }

    public boolean isPackedBlockAttentionEnabled() {
        return packedBlockAttentionEnabled;
    }

    public boolean isPackedPrefillEnabled() {
        return packedPrefillEnabled;
    }

    public boolean isTensorProviderExplicit() {
        return tensorProviderExplicit;
    }

    public Optional<TensorOperations> tensorOperations(TensorProviderKind kind) {
        return Optional.ofNullable(tensorOperations.get(kind));
    }

    String tensorOperationsSummary() {
        return tensorOperations.entrySet().stream()
                .sorted(Map.Entry.comparingByKey())
                .map(entry -> entry.getKey() + "=" + entry.getValue().name()
                        + "(parallelSplitSize=" + entry.getValue().parallelSplitSize() + ")")
                .collect(Collectors.joining(", "));
    }

    public TensorOperations primaryTensorOperations() {
        return configurableTensorProvider.get();
    }

    public TensorOperations prefillProjectionOperations(AbstractTensor input, AbstractTensor weight,
            io.teknek.deliverance.generator.ForwardPhase phase) {
        boolean useGpu = (gpuPrefillEnabled && phase == io.teknek.deliverance.generator.ForwardPhase.PREFILL
                && input.shape().first() >= 384)
                || (gpuDecodeEnabled && phase == io.teknek.deliverance.generator.ForwardPhase.DECODE
                && input.shape().first() == 1);
        if (useGpu
                && !tensorProviderExplicit
                && (input.dType() == DType.F32 || input.dType() == DType.I8)
                && (weight.dType() == DType.F32 || weight.dType() == DType.BF16 || weight.dType() == DType.Q4)) {
            Optional<TensorOperations> gpu = tensorOperations(TensorProviderKind.GPU);
            if (gpu.isPresent()) {
                TensorOperations operations = gpu.get();
                operations.registerModelTensor(weight);
                if (InferenceProfiler.isEnabled()) {
                    String phaseName = phase == io.teknek.deliverance.generator.ForwardPhase.PREFILL ? "prefill" : "decode";
                    InferenceProfiler.counter(metricRegistry, phaseName + ".projection_provider_gpu").inc();
                }
                return operations;
            }
        }
        return primaryTensorOperations();
    }

    public TensorParallelContext getTensorParallelContext() {
        return tensorParallelContext;
    }

    public TensorParallelCollectives getTensorParallelCollectives() {
        return tensorParallelCollectives;
    }

    public KvBufferCache.KvBuffer newKvBuffer() {
        return kvBufferCache.getEphemeralKvBuffer();
    }

    public KvCacheManager kvCacheManager() {
        return kvCacheManager;
    }

    public KvCacheSession newKvCacheSession() {
        return kvCacheManager.openSession();
    }

    public int restorePrefixToKvBuffer(int[] promptTokens, Optional<String> cacheSalt,
            KvBufferCache.KvBuffer destination) {
        KvBufferCache.PrefixEntry prefixHit = kvBufferCache.lookupPrefix(promptTokens, cacheSalt);
        if (prefixHit == null) {
            return 0;
        }
        try {
            kvBufferCache.copyPrefix(prefixHit.buffer(), destination, prefixHit.length());
            generationDebugHook.accept(new GenerationDebugEvent(
                    GenerationDebugEventType.AFTER_PREFIX_COPY,
                    promptTokens,
                    prefixHit.length(),
                    prefixHit.length(),
                    promptTokens.length - prefixHit.length(),
                    destination));
            return prefixHit.length();
        } finally {
            prefixHit.closeIfTemporary();
        }
    }

    public void storePrefixFromKvBuffer(int[] promptTokens, KvBufferCache.KvBuffer source, Optional<String> cacheSalt) {
        kvBufferCache.storePrefix(promptTokens, source, cacheSalt);
    }

    public void emitPromptPrefillDebug(int[] promptTokens, int prefixLength, int startPosition,
            int tokensToProcessLength, KvBufferCache.KvBuffer kvBuffer) {
        generationDebugHook.accept(new GenerationDebugEvent(
                GenerationDebugEventType.AFTER_PROMPT_PREFILL,
                promptTokens,
                prefixLength,
                startPosition,
                tokensToProcessLength,
                kvBuffer));
    }

    public int getLocalNumberOfHeads() {
        return config.numberOfHeads / tensorParallelContext.size();
    }

    public int getLocalNumberOfKeyValueHeads() {
        return config.numberOfKeyValueHeads / tensorParallelContext.size();
    }

    public int getLocalAttentionLength() {
        return config.attentionLength / tensorParallelContext.size();
    }

    public int getLocalKvLength() {
        return config.kvLength / tensorParallelContext.size();
    }

    /**
     * Optional model-family hook for RoPE variants not represented by {@link Config#ropeFreqs}.
     *
     * <p>The default returns {@code false}, allowing {@code CausalSelfAttention} to use its existing configured RoPE path.
     * Models with custom RoPE, such as YaRN plus model-specific query scaling, can mutate {@code query} and {@code key}
     * in place and return {@code true}.</p>
     */
    public boolean applyRotaryEmbedding(AbstractTensor query, AbstractTensor key, int absolutePosition,
            int queryHeads, int keyValueHeads, int headSize, TensorOperations operations) {
        return false;
    }

    /**
     * Installs a transient observer for generation internals.
     *
     * <p>This hook is intentionally diagnostic rather than API-facing. It exists so tests and local debugging can
     * inspect prefix-cache control flow or compute immediate KV fingerprints without sprinkling temporary printlns
     * through generation. The callback must not retain references to tensors or KV buffers; compute any diagnostics
     * inside the callback while the event is being delivered.</p>
     */
    public void setGenerationDebugHook(Consumer<GenerationDebugEvent> generationDebugHook) {
        this.generationDebugHook = generationDebugHook == null ? event -> {} : generationDebugHook;
    }

    public void clearGenerationDebugHook() {
        this.generationDebugHook = event -> {};
    }

    public void setLayerDebugHook(Consumer<LayerDebugEvent> layerDebugHook) {
        this.layerDebugHook = layerDebugHook == null ? event -> {} : layerDebugHook;
    }

    public void clearLayerDebugHook() {
        this.layerDebugHook = event -> {};
    }

    public void emitLayerDebug(int layerIndex, String stage, AbstractTensor hiddenStates) {
        layerDebugHook.accept(new LayerDebugEvent(layerIndex, stage, tensorParallelContext, hiddenStates));
    }

    void emitGenerationDebug(GenerationDebugEvent event) {
        generationDebugHook.accept(event);
    }

    private final ConcurrentHashMap<String, ResolvedLoraAdapter> registeredLoraAdapters = new ConcurrentHashMap<>();
    private final AtomicReference<ResolvedLoraAdapter> activeLoraAdapter = new AtomicReference<>();
    private final AtomicReference<String> activeLoraAdapterId = new AtomicReference<>();

    /**
     * Whether this model family supports LoRA runtime hot-swap ({@link #registerLoraAdapter}/
     * {@link #setActiveAdapter}).
     *
     * <p>Opt-in, not opt-out: defaults to {@code false}, matching {@link WeightLoader}'s own
     * "must opt in by overriding" convention for operations that aren't safe/meaningful
     * everywhere. Families whose {@code loadTransformerBlockWeights()} threads real per-layer base
     * tensor names through plain {@code CausalSelfAttention}/{@code MLPBlock} construction may
     * override this to {@code true}. Families with MoE-routed feed-forward layers (one base tensor
     * name per expert, not per layer) or independent, non-shared forward-pass classes must not --
     * a LoRA delta silently applied to only part of a targeted module's projections would be a
     * confusing, silently-partial result. See step 4 plan Section 6.</p>
     */
    protected boolean supportsLoraHotSwap() {
        return false;
    }

    /**
     * Registers a LoRA adapter for runtime hot-swap under {@code adapterId}, without activating
     * it. Call {@link #setActiveAdapter(String)} to actually apply it during generation.
     */
    public void registerLoraAdapter(String adapterId, LoraAdapter adapter) {
        if (!supportsLoraHotSwap()) {
            throw new UnsupportedOperationException(getClass().getSimpleName()
                    + " does not support LoRA runtime hot-swap -- see step 4 plan Section 6");
        }
        registeredLoraAdapters.put(adapterId, new ResolvedLoraAdapter(adapter, getWorkingDType()));
    }

    public void registerLoraAdapter(String adapterId, LoraAdapterModelFetcher fetcher) {
        registerLoraAdapter(adapterId, LoraAdapter.fromPretrained(fetcher, metricRegistry));
    }

    /** Unregisters and closes a previously-registered adapter. It must not be the active adapter. */
    public void unregisterLoraAdapter(String adapterId) {
        if (activeLoraAdapter.get() == registeredLoraAdapters.get(adapterId)) {
            throw new IllegalStateException("Cannot unregister the currently active LoRA adapter \""
                    + adapterId + "\" -- call clearActiveAdapter() first");
        }
        ResolvedLoraAdapter removed = registeredLoraAdapters.remove(adapterId);
        if (removed != null) {
            removed.close();
        }
    }

    /** Activates a previously-registered LoRA adapter; every subsequent forward pass applies its deltas. */
    public void setActiveAdapter(String adapterId) {
        if (tensorParallelContext.enabled()) {
            throw new UnsupportedOperationException(
                    "LoRA runtime hot-swap does not support tensor-parallel inference -- see step 4 plan Section 7");
        }
        ResolvedLoraAdapter resolved = registeredLoraAdapters.get(adapterId);
        if (resolved == null) {
            throw new IllegalArgumentException("No LoRA adapter registered under id \"" + adapterId + "\"");
        }
        activeLoraAdapter.set(resolved);
        activeLoraAdapterId.set(adapterId);
    }

    /** Deactivates the currently active LoRA adapter, if any, restoring plain base-model behavior. */
    public void clearActiveAdapter() {
        activeLoraAdapter.set(null);
        activeLoraAdapterId.set(null);
    }

    /**
     * Returns the currently active LoRA adapter's registered id, or empty if none is active.
     *
     * <p>Used to scope the local prefix cache per active adapter (see {@link
     * io.teknek.deliverance.model.LocalGenerationBackend}) -- KV state computed under one adapter
     * (or none) must never be reused for a request running under a different adapter, even when
     * the prompt tokens are byte-identical, or a cache hit would silently apply the wrong (or no)
     * adapter's effect to the cached prefix. See step 4 plan Section 11 item 12.</p>
     */
    public Optional<String> activeLoraAdapterId() {
        return Optional.ofNullable(activeLoraAdapterId.get());
    }

    /**
     * Returns the active adapter's resolved delta for {@code baseTensorName}, or empty if no
     * adapter is active or the active adapter doesn't target that tensor. Called once per targeted
     * projection per {@code forward()} call by {@code CausalSelfAttention}/{@code MLPBlock}.
     */
    public Optional<LoraLayerDelta> activeLoraDeltaFor(String baseTensorName) {
        ResolvedLoraAdapter active = activeLoraAdapter.get();
        return active == null ? Optional.empty() : active.deltaFor(baseTensorName);
    }

    /**
     * Forces the model's disk-backed KV page cleanup pass to run immediately.
     *
     * <p>This is a maintenance/test hook for active disk KV page storage. It does not operate on prefix-cache entries or
     * any persistent token-prefix manifest.</p>
     */
    public void runDiskKvPageSweep() {
        kvBufferCache.runDiskPageSweep();
    }

    /**
     * Returns the primary input kind accepted by this model at its forward/generation boundary.
     *
     * <p>Text generation models normally use {@link ModelInputName#INPUT_IDS}. Models with non-text primary inputs,
     * such as audio or vision encoders, should override this method with the input kind they expect. This is a runtime
     * model capability, not a raw checkpoint configuration value.</p>
     */
    public ModelInputName getMainInputName() {
        return ModelInputName.INPUT_IDS;
    }

    /**
     * Prepares the typed model inputs for one generation forward step.
     *
     * @param inputIds token ids for one request. When used, this is a one-dimensional sequence of vocabulary ids.
     * @param inputsEmbeds optional precomputed embeddings with shape {@code [batch, sequence, embedding]}. Dimension 0
     * is request batch index, dimension 1 is token position, and dimension 2 is the dense embedding vector.
     * @param attentionMask optional one-dimensional mask aligned to the sequence dimension; non-null values are sliced
     * to match the prepared sequence length.
     * @param encoderAttentionMask optional one-dimensional encoder-side mask for encoder-decoder models. It is retained
     * as encoder input context and is not sliced to the decoder sequence length.
     * @param positionIds optional one-dimensional positions aligned to the sequence dimension; non-null values are
     * sliced to match the prepared sequence length.
     * @param tokenTypeIds optional one-dimensional segment ids aligned to the sequence dimension; non-null values are
     * sliced to match the prepared sequence length.
     * @param mmTokenTypeIds optional one-dimensional multimodal token type ids aligned to the sequence dimension;
     * non-null values are sliced to match the prepared sequence length.
     */
    protected GenerationStepInputs prepareInputsForGeneration(int[] inputIds, Integer nextSequenceLength, PastKeyValues pastKeyValues, int[] attentionMask, int[] encoderAttentionMask, AbstractTensor inputsEmbeds, boolean firstIteration, int[] positionIds, int[] tokenTypeIds, int[] mmTokenTypeIds) {
        return GenerationInputPreparer.prepareInputsForGeneration(config, inputIds, nextSequenceLength, pastKeyValues, attentionMask, encoderAttentionMask, inputsEmbeds, firstIteration, positionIds, tokenTypeIds, mmTokenTypeIds, this::makeDenseTensor);
    }

    protected abstract EmbedInput loadInputWeights();
    protected abstract SampleOutput loadOutputWeights();
    protected abstract TransformerBlock[] loadTransformerBlockWeights();

    @Override
    public void close() {
        registeredLoraAdapters.values().forEach(ResolvedLoraAdapter::close);
        registeredLoraAdapters.clear();
        activeLoraAdapter.set(null);
        activeLoraAdapterId.set(null);
        if (gossipParallelMembership != null) {
            gossipParallelMembership.close();
            gossipParallelMembership = null;
        }
        if (tensorParallelCollectives instanceof AutoCloseable closeable) {
            try {
                closeable.close();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
        kvBufferCache.close();
        closeTensorOperations();
        try {
            weights.close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private void closeTensorOperations() {
        Set<TensorOperations> closed = Collections.newSetFromMap(new IdentityHashMap<>());
        TensorOperations primary = configurableTensorProvider.get();
        if (closed.add(primary)) {
            primary.close();
        }
        for (TensorOperations operations : tensorOperations.values()) {
            if (closed.add(operations)) {
                operations.close();
            }
        }
    }

    public Optional<GossipParallelMembership> gossipParallelMembership() {
        return Optional.ofNullable(gossipParallelMembership);
    }

    void setMaxBatchSize(int maxBatchSize) {
        if (maxBatchSize < 1) {
            throw new IllegalArgumentException("maxBatchSize must be >= 1");
        }
        this.maxBatchSize = maxBatchSize;
    }

    void setGossipParallelMembership(GossipParallelMembership gossipParallelMembership) {
        this.gossipParallelMembership = Objects.requireNonNull(gossipParallelMembership, "gossipParallelMembership");
    }

    public Config getConfig(){
        return config;
    }

    public AbstractTensor makeTensor(int... shape) {
        TensorShape s = TensorShape.of(shape);
        return tensorAllocator.get(workingDType, s);
    }

    public AbstractTensor makeDenseTensor(int... shape) {
        return tensorAllocator.get(workingDType, TensorShape.of(shape));
    }

    public AbstractTensor makeDenseTensor(TensorShape s) {
        return tensorAllocator.get(workingDType, s);
    }

    public DType getWorkingDType() {
        return workingDType;
    }

    public DType getWorkingQType() {
        return workingQType;
    }

    /**
     * Returns whether a tensor is already in this model's working quantized dtype.
     *
     * <p>This method exists to keep tensor ownership explicit at call sites. Older code used {@code maybeQuantize(...)},
     * which can return different temporary/copy forms depending on dtype and makes close behavior hard to reason about.
     * New code should branch on this predicate: use the original tensor directly when it is already in the desired dtype,
     * or call {@link #quantizeToWorkingQuantizedType(AbstractTensor)} when a new temporary tensor is required.</p>
     */
    public boolean isInWorkingQuantizedType(AbstractTensor tensor) {
        return tensor.dType() == workingQType;
    }

    /**
     * Quantizes a tensor to this model's working quantized dtype and always returns a new caller-owned tensor.
     *
     * <p>This method deliberately does not return the input tensor even if it already has the target dtype. Callers that
     * want to avoid an unnecessary temporary must first check {@link #isInWorkingQuantizedType(AbstractTensor)} and use the
     * original tensor in that branch. This keeps resource ownership visible: tensors returned by this method must be
     * closed by the caller.</p>
     */
    public AbstractTensor quantizeToWorkingQuantizedType(AbstractTensor tensor) {
        return configurableTensorProvider.get().quantize(tensor, workingQType, 0,
                Math.toIntExact(tensor.shape().last()));
    }

    public DType getModelDType() {
        return modelDType;
    }

    public String getTensorProviderName() {
        return configurableTensorProvider.get().name();
    }

    public int getTensorProviderParallelSplitSize() {
        return configurableTensorProvider.get().parallelSplitSize();
    }

    /**
     *
     * @return Some if the tokenizer inside this model has a chat_template/prompt template Empty if not.
     */
    public Optional<PromptSupport> promptSupport() {
        return tokenizer.chatTemplate().map(template -> new PromptSupport(
                Map.of("default", template),
                tokenizer.bosToken().orElse(""),
                tokenizer.eosToken().orElse(""),
                template.toLowerCase(Locale.ROOT).contains("tools")));
    }

    protected long[] encodeText(String text) {
        return Arrays.stream(tokenizer.encode(text, EncodeOptions.defaults().withoutSpecialTokens()).inputIds()).asLongStream().toArray();
    }

    /**
     * Exposes the actual runtime prompt-token encoding path for debugging and tests.
     */
    public long[] encodeForRuntime(String text) {
        return encodeText(text);
    }

    /**
     * Exposes the final generation prompt-token construction path, including any BOS insertion.
     */
    public int[] constructPromptTokensForRuntime(String text) {
        return constructPromptTokens(encodeText(text));
    }

    protected String decodeToken(long token) {
        return tokenizer.decode(new io.teknek.deliverance.grace.TokenIds(Ints.checkedCast(token)), false, false, false, false);
    }

    public String decodeToken(int token) {
        return decodeToken((long) token);
    }

    protected boolean addBosToken() {
        return true;
    }

    /**
     *
     * @return an array with bos token appened at the beginning if the model calls for it
     */
    int [] constructPromptTokens(long[] encoded){
        int[] promptTokens;
        if (addBosToken()) {
            promptTokens = new int[(1 + encoded.length)];
            promptTokens[0] = config.bosToken;
            for (int i = 1; i <= encoded.length; i++) {
                promptTokens[i] = Ints.checkedCast(encoded[i - 1]);
            }
        } else {
            promptTokens = Arrays.stream(encoded).mapToInt(Ints::checkedCast).toArray();
        }
        return promptTokens;
    }

    SamplerReturn createNextToken(GeneratorParameters generatorParameters, GenerationEngine.Logits logits, GenerationEngine.PrefillOutput last,
                                   ResponseContext responseContext, Random random, float temperature,
                                  Optional<LogitsProcessor> logitsProcessor, AbstractTensor argMaxScratch){
        try (AbstractTensor lastTokenOutput = last.copyLastTokenOutput(tensorAllocator)) {
            DeliveranceSampler legacy = new DeliveranceSampler(this, generatorParameters,
                    lastTokenOutput, logits.tensor(), sampleOutput.getOutputLayerNorm(), random, random.nextFloat(),
                    responseContext, argMaxScratch, logitsProcessor);
            return legacy.sample();
        }
    }

    SamplerReturn createNextTokenLoop(GeneratorParameters generatorParameters, AbstractTensor output,
                             AbstractTensor logits, ResponseContext responseContext, Random random, float temperature,
                             Optional<LogitsProcessor> logitsProcessor, AbstractTensor argMaxScratch){
        DeliveranceSampler legacy = new DeliveranceSampler(this, generatorParameters, output, logits,
                sampleOutput.getOutputLayerNorm(), random, random.nextFloat(), responseContext, argMaxScratch,
                logitsProcessor);
        return legacy.sample();
    }

    /**
     * S
     * @return Some if request should terminate None to continue
     */
    public Optional<Response> stopWords(GeneratorParameters generatorParameters, ResponseContext responseContext, int promptLength) {
        if (generatorParameters.stopWords.isPresent()){
            List<String> stops = generatorParameters.stopWords.get();
            for (String stop: stops){
                if (responseContext.responseTextWithSpecialTokens.indexOf(stop) != -1) {
                    FinishReason reason = FinishReason.STOP_TOKEN;
                    if (generatorParameters.includeStopStrInOutput.isPresent() && generatorParameters.includeStopStrInOutput.get()){
                        return Optional.of(new Response(responseContext.responseText.toString(), responseContext.responseTextWithSpecialTokens.toString(),
                                reason, promptLength, responseContext.generatedTokens, 0, 0,
                                responseContext.samplerReturnList));
                    } else {
                        int index = responseContext.responseTextWithSpecialTokens.indexOf(stop);
                        responseContext.responseTextWithSpecialTokens.delete(index, responseContext.responseTextWithSpecialTokens.length());
                        int x = responseContext.responseText.indexOf(stop);
                        if (x != -1) {
                            responseContext.responseText.delete(x, responseContext.responseText.length());
                        }
                        return Optional.of(new Response(responseContext.responseText.toString(), responseContext.responseTextWithSpecialTokens.toString(),
                                reason, promptLength, responseContext.generatedTokens, 0, 0,
                                responseContext.samplerReturnList));
                    }
                }
            }
        }
        return Optional.empty();
    }

    protected Response postProcessResponse(Response response) {
        return response;
    }

    /** Returns true when a model family owns generation semantics that should bypass the generic AR engine. */
    public boolean usesModelSpecificGeneration() {
        return false;
    }

    @Override
    public Response generate(UUID sessionId, PromptContext promptContext, GeneratorParameters generatorParameters,
                                       GenerateEvent eventFired) {
        return DefaultCausalLanguageModel.local(this).generate(sessionId, promptContext, generatorParameters, eventFired);
    }

    /**
     * Runs the standard generation/token sampling loop while delegating transformer forward execution.
     *
     * <p>This is used by tensor-parallel coordinators: the coordinator model still owns tokenizer, output projection,
     * sampler, stop handling, and response post-processing, while rank endpoints own prompt/decode forward execution and
     * KV state. Prefix-cache reuse is intentionally local to {@link #generate(UUID, PromptContext, GeneratorParameters,
     * GenerateEvent)} because this method's KV state lives behind the supplied forwarder.</p>
     */
    public Response generateWithForwarder(UUID sessionId, PromptContext promptContext, GeneratorParameters generatorParameters,
                                           GenerateEvent eventFired, GenerationForwarder forwarder) {
        Objects.requireNonNull(sessionId, "sessionId");
        Objects.requireNonNull(forwarder, "forwarder");
        return new GenerationEngine().generate(this, new ForwarderGenerationBackend(forwarder), sessionId, promptContext,
                generatorParameters, eventFired);
    }

    public Response generateWithBackend(UUID sessionId, PromptContext promptContext, GeneratorParameters generatorParameters,
            GenerateEvent eventFired, GenerationBackend backend) {
        Objects.requireNonNull(sessionId, "sessionId");
        Objects.requireNonNull(backend, "backend");
        return new GenerationEngine().generate(this, backend, sessionId, promptContext, generatorParameters, eventFired);
    }

    @Override
    public SortedMap<String, Float> classify(String input, PoolingType poolingType) {
        if (!config.isClassifier()) {
            throw new UnsupportedOperationException("Classification not supported by this model");
        }
        if (this.classifyOutput == null){
            throw new UnsupportedOperationException("classifyOutput was not setup");
        }
        float[] embedding = embed(input, poolingType);
        FloatBufferTensor b = new FloatBufferTensor(FloatBuffer.wrap(embedding), TensorShape.of(embedding.length), false);
        int classes = classifyOutput.getClassificationWeights().shape().first();
        //AbstractTensor scores = makeDenseTensor(classes);
        AbstractTensor scores = tensorAllocator.getDirty(workingDType, TensorShape.of(classes));
        metricRegistry.timer("classify.1_dotproduct_scores").time(() ->
            configurableTensorProvider.get().batchDotProduct(scores, b, classifyOutput.getClassificationWeights(),
                0, 0, config.embeddingLength));
        metricRegistry.timer("classify.2_accumulate_scores_bias").time(() ->
        classifyOutput.getClassificationBias().ifPresent(bias ->
                configurableTensorProvider.get().accumulate(scores, bias, 0, classes)) );
        metricRegistry.timer("classify.3_softmax_scores").time(() ->
        configurableTensorProvider.get().softMax(scores, 0, classes));
        SortedMap<String, Float> result = new TreeMap<>();
        for (int i = 0; i < classes; i++) {
            String label = config.classifcationLabels.get().inverse().get(i);
            Float score = scores.get(0, i);
            result.put(label, score);
        }
        return result;
    }

    public float[] embed(String input, PoolingType poolingType) {
        //TODO better recipe then this? timed callable
        Timer.Context c = metricRegistry.timer("abstractmodel.embed").time();
        try {
            return timedEmbedding(input, poolingType);
        } finally {
            c.stop();
        }
    }
    protected float[] timedEmbedding(String input, PoolingType poolingType) {
        CausualWhisperer.LOGGER.debug("embedding on {} using pooling type {}", input, poolingType);
        int[] encoded = Arrays.stream(encodeText(input)).mapToInt(Ints::checkedCast).toArray();
        Preconditions.checkArgument(encoded.length < config.contextLength);
        float [] outputEmbedding = new float[config.embeddingLength];
        CausualWhisperer.LOGGER.debug("created float [] outputEmbedding of length {}", config.embeddingLength);

        try (KvBufferCache.KvBuffer kvMem = kvBufferCache.getEphemeralKvBuffer()){
            int promptLength = encoded.length;
            float avgp = 1.0f / promptLength;
            CausualWhisperer.LOGGER.debug("1.0f / promptLength {} = avgp {}", promptLength, avgp);

            try (AbstractTensor r = metricRegistry.timer("abstractmodel.embed_1_batchforward").timeSupplier(()
                    -> batchForward(encoded, 0, kvMem))){
                if (poolingType == PoolingType.MODEL){
                    if (poolingLayer.isEmpty()){
                        throw new UnsupportedOperationException("no pooling layer for this model");
                    }
                    AbstractTensor output = r.slice(promptLength - 1);
                    //AbstractTensor pooled = makeDenseTensor(1, config.embeddingLength);
                    AbstractTensor pooled = tensorAllocator.getDirty(workingDType, TensorShape.of(1, config.embeddingLength));
                    configurableTensorProvider.get()
                            .batchDotProduct(pooled, output, poolingLayer.get().getPoolingWeights(), 0, 0, config.embeddingLength);
                    poolingLayer.get()
                            .getPoolingBias()
                            .ifPresent(bias -> { configurableTensorProvider.get().accumulate(pooled, bias, 0, config.embeddingLength); });
                    VectorMath.pfor(0, config.embeddingLength, i -> {
                        // BERT seems to use tanh for pooling rather than gelu
                        //outputEmbedding[i] = ActivationFunction.eval(ActivationFunction.Type.TANH, pooled.get(0, i));
                        outputEmbedding[i] = ActivationFunction.eval(config.activationFunction, pooled.get(0, i));
                    }, pool);
                    return outputEmbedding;
                }
                for (int i = 0; i < promptLength; i++) {
                    AbstractTensor output = r.slice(i);
                    // Pooling
                    for (int ii = 0; ii < config.embeddingLength; ii++) {

                        switch (poolingType) {
                            case AVG:
                                outputEmbedding[ii] += output.get(0, ii) * avgp;
                                break;
                            case MAX:
                                outputEmbedding[ii] = Math.max(outputEmbedding[ii], output.get(0, ii));
                                break;
                            case SUM:
                                outputEmbedding[ii] += output.get(0, ii);
                                break;
                        }
                    }
                }
                VectorMathUtils.l2normalize(outputEmbedding);
                return outputEmbedding;
            }
        }
    }


    public AbstractTensor batchForward(int[] token_ids, int startPos, KvBufferCache.KvBuffer kvbuf) {
        return batchForward(token_ids, startPos, kvbuf, Optional.empty());
    }

    public AbstractTensor batchForward(int[] tokenIds, int startPos) {
        try (KvBufferCache.KvBuffer kvBuffer = kvBufferCache.getEphemeralKvBuffer()) {
            return batchForward(tokenIds, startPos, kvBuffer, Optional.empty());
        }
    }

    public AbstractTensor batchForward(int[] token_ids, int startPos, KvBufferCache.KvBuffer kvbuf,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, "abstractmodel.batch_forward").time()) {
            AbstractTensor lastBatchOutput = null;

            CausualWhisperer.LOGGER.debug("batchForward from 0 to token_ids.length {} max_batch_size {} per iteration",
                    token_ids.length, maxBatchSize);
            PrefillProgress previousProgress = PREFILL_PROGRESS.get();
            PrefillProgress progress = new PrefillProgress(token_ids.length, startPos, System.nanoTime());
            PREFILL_PROGRESS.set(progress);
            try {
                for (int i = 0; i < token_ids.length; i += maxBatchSize) {
                    throwIfGenerationInterrupted();
                    int[] batch = Arrays.copyOfRange(token_ids, i, Math.min(token_ids.length, i + maxBatchSize));
                    progress.chunkStart = i;
                    progress.chunkTokens = batch.length;
                    AbstractTensor inputEmbeddings = embedInput.batchInputsToEmbeddings(batch, startPos + i);
                    PlannedTensor plannedEmbeddings = plannedInputEmbeddings("input_embeddings", inputEmbeddings,
                            io.teknek.deliverance.generator.ForwardPhase.PREFILL);
                    lastBatchOutput = forward(plannedEmbeddings, startPos + i, kvbuf, tensorReducer,
                            io.teknek.deliverance.generator.ForwardPhase.PREFILL).tensor();
                    int processed = Math.min(token_ids.length, i + batch.length);
                    long now = System.nanoTime();
                    if (processed < token_ids.length && now >= progress.nextLogNanos) {
                        logPrefillProgress(progress, progress.chunkStart, config.numberOfLayers, config.numberOfLayers, now);
                        progress.nextLogNanos = now + PREFILL_PROGRESS_INTERVAL_NANOS;
                    }
                }
            } finally {
                if (previousProgress == null) {
                    PREFILL_PROGRESS.remove();
                } else {
                    PREFILL_PROGRESS.set(previousProgress);
                }
            }
            return lastBatchOutput;
        }
    }

    public AbstractTensor forward(int token_id, int pos, KvBufferCache.KvBuffer kvbuf) {
        return forward(token_id, pos, kvbuf, Optional.empty());
    }

    /**
     * This is a distributed version of forward pass that serves as a coordination point for the
     * distributed model.  The layers are split into one or more heads and each head is processed
     * by a different node.
     *
     * @param token_id
     * @param pos
     * @param kvbuf
     * @return
     */
    public AbstractTensor forward(int token_id, int pos, KvBufferCache.KvBuffer kvbuf,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, "abstractmodel.forward_token").time()) {
        AbstractTensor embedding = embedInput.inputTokenToEmbedding(token_id, pos);
        return forward(plannedInputEmbeddings("input_embedding", embedding,
                io.teknek.deliverance.generator.ForwardPhase.DECODE), pos, kvbuf, tensorReducer,
                io.teknek.deliverance.generator.ForwardPhase.DECODE).tensor();
        }
    }


    public AbstractTensor forward(AbstractTensor embedding, int startPos, KvBufferCache.KvBuffer kvbuf,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        return forward(embedding, startPos, kvbuf, tensorReducer, io.teknek.deliverance.generator.ForwardPhase.DECODE);
    }

    public AbstractTensor forward(AbstractTensor embedding, int startPos, KvBufferCache.KvBuffer kvbuf,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer, io.teknek.deliverance.generator.ForwardPhase phase) {
        PlannedTensor planned = new PlannedTensor(embedding,
                modelLineagePlan.input("forward.input", embedding).as("forward.input"));
        return forward(planned, startPos, kvbuf, tensorReducer, phase).tensor();
    }

    private PlannedTensor plannedInputEmbeddings(String name, AbstractTensor embeddings,
            io.teknek.deliverance.generator.ForwardPhase phase) {
        TensorPlan.Tensor lineage = modelLineageTensor("model.embed_tokens.weight")
                .map(upstream -> modelLineagePlan.input(name, upstream, embeddings))
                .orElseGet(() -> modelLineagePlan.input(name, embeddings))
                .as(name);
        traceTensorPlan(getClass().getSimpleName(), "embedinput." + name, phase.name(), -1, "N/A", lineage.plan());
        return new PlannedTensor(embeddings, lineage);
    }

    public PlannedTensor forward(PlannedTensor embedding, int startPos, KvBufferCache.KvBuffer kvbuf,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer, io.teknek.deliverance.generator.ForwardPhase phase) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, "abstractmodel.forward_layers").time()) {
        emitLayerDebug(-1, "input", embedding.tensor());
        int batchTokens = embedding.tensor().shape().first();
        for (int i = 0; i < config.numberOfLayers; i++) {
            throwIfGenerationInterrupted();
            int relativeLayer = i;
            AbstractTensor ref = embedding.tensor(); // reference so we can free
            embedding = transformerBlocks[relativeLayer].forward(embedding, startPos, kvbuf, tensorReducer, phase);
            emitLayerDebug(relativeLayer, "layer_output", embedding.tensor());
            ref.close();
            long now = System.nanoTime();
            PrefillProgress progress = PREFILL_PROGRESS.get();
            if (progress != null && batchTokens > 1 && i + 1 < config.numberOfLayers && now >= progress.nextLogNanos) {
                logPrefillProgress(progress, progress.chunkStart, i + 1, config.numberOfLayers, now);
                progress.nextLogNanos = now + PREFILL_PROGRESS_INTERVAL_NANOS;
            }
        }
        return embedding;
        }
    }

    private static void throwIfGenerationInterrupted() {
        if (Thread.interrupted()) {
            throw new CancellationException("generation interrupted");
        }
    }

    private static void logPrefillProgress(PrefillProgress progress, int completedTokensBeforeCurrentChunk,
            int processedLayers, int totalLayers, long now) {
        int estimatedCurrentChunkTokens = totalLayers == 0 ? 0 : (progress.chunkTokens * processedLayers) / totalLayers;
        int estimatedProcessedTokens = Math.min(progress.totalTokens,
                completedTokensBeforeCurrentChunk + estimatedCurrentChunkTokens);
        double elapsedSeconds = (now - progress.startNanos) / 1_000_000_000.0;
        double tokensPerSecond = elapsedSeconds == 0.0 ? 0.0 : estimatedProcessedTokens / elapsedSeconds;
        double remainingSeconds = tokensPerSecond == 0.0
                ? Double.NaN
                : (progress.totalTokens - estimatedProcessedTokens) / tokensPerSecond;
        int chunkStartPosition = progress.startPos + progress.chunkStart;
        int chunkEndPosition = chunkStartPosition + progress.chunkTokens - 1;
        logger.info("prefill progress tokens={}/{} chunk={}-{} layers={}/{} elapsed={} eta={} rate={} tok/s",
                estimatedProcessedTokens,
                progress.totalTokens,
                chunkStartPosition,
                chunkEndPosition,
                processedLayers,
                totalLayers,
                seconds(elapsedSeconds),
                seconds(remainingSeconds),
                rate(tokensPerSecond));
    }

    private static String seconds(double seconds) {
        if (!Double.isFinite(seconds)) {
            return "unknown";
        }
        return String.format(Locale.ROOT, "%.1fs", seconds);
    }

    private static String rate(double tokensPerSecond) {
        if (!Double.isFinite(tokensPerSecond)) {
            return "unknown";
        }
        return String.format(Locale.ROOT, "%.1f", tokensPerSecond);
    }

    /** Returns an owned working tensor that callers may safely close. */
    public AbstractTensor maybeQuantize(AbstractTensor t) {
        AbstractTensor t2 = tensorAllocator.getDirty(t.dType(), t.shape());
        t2.copyFrom(t, 0, 0, Ints.checkedCast(t.size()));
        return t2;
    }

    /** Returns a close-safe read-only tensor when no dtype conversion is needed; otherwise returns an owned tensor. */
    public AbstractTensor maybeQuantizeReadOnly(AbstractTensor t, String counterPrefix) {
        if (t.dType() == workingQType) {
            InferenceProfiler.counter(metricRegistry, counterPrefix + ".read_only").inc();
            return new ReadOnlyTensor(t);
        }
        InferenceProfiler.counter(metricRegistry, counterPrefix + ".copy_or_quantize").inc();
        return quantizeToWorkingQuantizedType(t);
    }

    public PreTrainedTokenizer getTokenizer(){
        return this.tokenizer;
    }

    public boolean isSpecialToken(int token) {
        return tokenizer.allSpecialIds().contains(token);
    }

    public TensorAllocator getTensorAllocator(){
        return tensorAllocator;
    }


    protected  ClassifyOutput loadClassifierWeights(){
        throw new IllegalArgumentException("loadClassifierWeights not yet implemented");
    }

    protected PoolingLayer loadPoolingWeights() {
        return null;
    }

    public ToolCallParser getToolCallParser() {
        return toolCallParser;
    }

    public MetricRegistry getMetricRegistry(){
        return metricRegistry;
    }

    public WrappedForkJoinPool getPool() {
        return pool;
    }

    public ConfigurableTensorProvider getConfigurableTensorProvider() {
        return configurableTensorProvider;
    }

    public void runChunks(String operation, int offset, int length, int splitSize, Optional<AbstractTensor> localityTensor,
            io.teknek.deliverance.math.BiIntConsumer action) {
        if (tensorRuntime != null) {
            tensorRuntime.runChunks(operation, offset, length, splitSize, localityTensor, action);
        } else {
            VectorMath.pchunk(offset, length, action, splitSize, pool);
        }
    }

    public Optional<TensorRuntimeMode> getTensorRuntimeMode() {
        return tensorRuntimeMode;
    }

    public void setTensorRuntimeMode(Optional<TensorRuntimeMode> tensorRuntimeMode) {
        this.tensorRuntimeMode = Objects.requireNonNull(tensorRuntimeMode, "tensorRuntimeMode");
    }

    public TensorRuntime getTensorRuntime() {
        return tensorRuntime;
    }

    public void setTensorRuntime(TensorRuntime tensorRuntime) {
        this.tensorRuntime = tensorRuntime;
    }

    public Map<String, Object> generationOptions() {
        return generationOptions;
    }

    void setGenerationOptions(Map<String, Object> generationOptions) {
        this.generationOptions = generationOptions == null ? Map.of() : Map.copyOf(generationOptions);
    }

    public void setTensorPlanTraceEnabled(boolean tensorPlanTraceEnabled) {
        this.tensorPlanTraceEnabled = tensorPlanTraceEnabled;
    }

    public TensorPlanTraceScope openTensorPlanTrace(UUID generationId) {
        if (!tensorPlanTraceEnabled) {
            return () -> {};
        }
        TensorPlanTraceContext context = new TensorPlanTraceContext(generationId, tensorPlanTraceHeader(generationId));
        tensorPlanTraces.put(generationId, context);
        activeTensorPlanTraceId.set(generationId);
        context.record(getClass().getSimpleName(), "model.weights", "INIT", -1, "N/A", renderModelLineagePlan());
        return () -> {
            activeTensorPlanTraceId.remove();
            TensorPlanTraceContext finished = tensorPlanTraces.remove(generationId);
            if (finished != null) {
                logger.info("\n{}", finished.render());
            }
        };
    }

    public void traceTensorPlan(String ownerClass, String path, String phase, int layerIndex, String runMode,
            String planText) {
        UUID generationId = activeTensorPlanTraceId.get();
        if (generationId == null) {
            return;
        }
        TensorPlanTraceContext context = tensorPlanTraces.get(generationId);
        if (context != null) {
            context.record(ownerClass, path, phase, layerIndex, runMode, planText);
        }
    }

    protected TensorPlan.ImmutableTensor registerModelLineageTensor(String name, AbstractTensor tensor) {
        if (modelLineagePlan == null) {
            return null;
        }
        TensorPlan.ImmutableTensor planned = modelLineagePlan.immutable(name, tensor);
        modelLineageTensors.put(name, planned);
        modelLineageEntries.add(new ModelLineageEntry(name, tensor.shape().toString(), tensor.dType().name(), planned.plan()));
        return planned;
    }

    public Optional<TensorPlan.ImmutableTensor> modelLineageTensor(String name) {
        return Optional.ofNullable(modelLineageTensors.get(name));
    }

    private String renderModelLineagePlan() {
        if (modelLineageEntries.isEmpty()) {
            return "└─ model.weights empty";
        }
        List<String> lines = groupedModelLineageLines();
        StringBuilder sb = new StringBuilder("└─ model.weights\n");
        for (int i = 0; i < lines.size(); i++) {
            boolean last = i + 1 == lines.size();
            sb.append(last ? "   └─ " : "   ├─ ")
                    .append(lines.get(i))
                    .append('\n');
        }
        return sb.toString();
    }

    private List<String> groupedModelLineageLines() {
        Map<ModelLineageGroupKey, ModelLineageGroup> groups = new LinkedHashMap<>();
        for (ModelLineageEntry entry : modelLineageEntries) {
            LayerName layerName = LayerName.parse(entry.name());
            ModelLineageGroupKey key = new ModelLineageGroupKey(layerName.pattern(), entry.shape(), entry.dtype());
            groups.computeIfAbsent(key, ignored -> new ModelLineageGroup(layerName.prefix(), layerName.suffix(),
                    entry.shape(), entry.dtype())).add(layerName.layer());
        }
        List<String> lines = new ArrayList<>();
        for (ModelLineageGroup group : groups.values()) {
            lines.add(group.render());
        }
        return lines;
    }

    private record ModelLineageEntry(String name, String shape, String dtype, String plan) {
    }

    private record ModelLineageGroupKey(String pattern, String shape, String dtype) {
    }

    private static final class ModelLineageGroup {
        private final String prefix;
        private final String suffix;
        private final String shape;
        private final String dtype;
        private final List<Integer> layers = new ArrayList<>();

        private ModelLineageGroup(String prefix, String suffix, String shape, String dtype) {
            this.prefix = prefix;
            this.suffix = suffix;
            this.shape = shape;
            this.dtype = dtype;
        }

        private void add(OptionalInt layer) {
            layer.ifPresent(layers::add);
        }

        private String render() {
            if (layers.isEmpty()) {
                return prefix + suffix + " " + shape + " " + dtype;
            }
            return prefix + "[" + ranges(layers) + "]" + suffix + " " + shape + " " + dtype
                    + " count=" + layers.size();
        }

        private static String ranges(List<Integer> values) {
            List<Integer> sorted = values.stream().distinct().sorted().toList();
            List<String> ranges = new ArrayList<>();
            int start = sorted.get(0);
            int previous = start;
            for (int i = 1; i < sorted.size(); i++) {
                int value = sorted.get(i);
                if (value == previous + 1) {
                    previous = value;
                    continue;
                }
                ranges.add(start == previous ? Integer.toString(start) : start + "-" + previous);
                start = previous = value;
            }
            ranges.add(start == previous ? Integer.toString(start) : start + "-" + previous);
            return String.join(",", ranges);
        }
    }

    private record LayerName(String prefix, OptionalInt layer, String suffix) {
        private String pattern() {
            return layer.isPresent() ? prefix + "{}" + suffix : prefix + suffix;
        }

        private static LayerName parse(String name) {
            String marker = "model.layers.";
            int markerIndex = name.indexOf(marker);
            if (markerIndex < 0) {
                return new LayerName(name, OptionalInt.empty(), "");
            }
            int start = markerIndex + marker.length();
            int end = start;
            while (end < name.length() && Character.isDigit(name.charAt(end))) {
                end++;
            }
            if (end == start || end >= name.length() || name.charAt(end) != '.') {
                return new LayerName(name, OptionalInt.empty(), "");
            }
            return new LayerName(name.substring(0, start), OptionalInt.of(Integer.parseInt(name.substring(start, end))),
                    name.substring(end));
        }
    }

    public interface TensorPlanTraceScope extends AutoCloseable {
        @Override
        void close();
    }

    private String tensorPlanTraceHeader(UUID generationId) {
        return "================ TensorPlan Trace ================\n"
                + "generationId=" + generationId + '\n'
                + "model.class=" + getClass().getSimpleName() + '\n'
                + "config.class=" + config.getClass().getSimpleName() + '\n'
                + "modelDType=" + modelDType + '\n'
                + "workingDType=" + workingDType + '\n'
                + "workingQType=" + workingQType + '\n'
                + "modelQType=" + modelQType + '\n'
                + "outputHeadQuantization=" + outputHeadQuantization + '\n'
                + "tensorProvider=" + configurableTensorProvider.get().name() + '\n'
                + "parallelSplitSize=" + configurableTensorProvider.get().parallelSplitSize() + '\n'
                + "tensorRuntimeMode=" + tensorRuntimeMode + '\n'
                + "gpuPrefill=" + gpuPrefillEnabled + '\n'
                + "gpuDecode=" + gpuDecodeEnabled + '\n'
                + "gpuDecodeAttention=" + gpuDecodeAttentionEnabled + '\n'
                + "gpuDiffusionBlockProjection=" + gpuDiffusionBlockProjectionEnabled + '\n'
                + "packedBlockAttention=" + packedBlockAttentionEnabled + '\n'
                + "packedPrefill=" + packedPrefillEnabled + '\n'
                + "generationOptions=" + generationOptions + '\n'
                + "tensorProviderExplicit=" + tensorProviderExplicit + '\n'
                + "layers=" + config.numberOfLayers + '\n'
                + "embeddingLength=" + config.embeddingLength + '\n'
                + "hiddenLength=" + config.hiddenLength + '\n'
                + "attentionLength=" + config.attentionLength + '\n'
                + "kvLength=" + config.kvLength + '\n'
                + "==================================================\n";
    }

    private static final class TensorPlanTraceContext {
        private final String header;
        private final Set<String> seen = ConcurrentHashMap.newKeySet();
        private final Queue<String> sections = new ConcurrentLinkedQueue<>();

        private TensorPlanTraceContext(UUID generationId, String header) {
            this.header = header;
        }

        private void record(String ownerClass, String path, String phase, int layerIndex, String runMode,
                String planText) {
            String key = ownerClass + '|' + path + '|' + phase + '|' + runMode + '|' + planText;
            if (!seen.add(key)) {
                return;
            }
            String layer = layerIndex < 0 ? "n/a" : Integer.toString(layerIndex);
            sections.add("[TensorPlan] owner=" + ownerClass + " path=" + path + " phase=" + phase
                    + " layer=" + layer + " runMode=" + runMode + '\n' + indent(planText) + '\n');
        }

        private String render() {
            StringBuilder sb = new StringBuilder(header);
            sections.forEach(sb::append);
            return sb.toString();
        }

        private static String indent(String text) {
            return text.lines().map(line -> "  " + line).reduce((a, b) -> a + '\n' + b).orElse("");
        }
    }

}
