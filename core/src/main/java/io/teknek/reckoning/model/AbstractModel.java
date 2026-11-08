package io.teknek.reckoning.model;

import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;

import java.util.*;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;


import io.teknek.reckoning.CausualWhisperer;
import io.teknek.reckoning.Classifier;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.classifier.ClassifyOutput;
import io.teknek.reckoning.embedding.PoolingLayer;

import io.teknek.reckoning.embedding.PoolingType;
import io.teknek.reckoning.generator.*;
import io.teknek.reckoning.generator2.SampleOutputRef;
import io.teknek.reckoning.guided.LogitsProcessor;
import io.teknek.reckoning.guided.LogitsProcessorRef;
import io.teknek.reckoning.grace.EncodeOptions;
import io.teknek.reckoning.grace.PreTrainedTokenizer;
import io.teknek.reckoning.math.ActivationFunction;
import io.teknek.reckoning.math.VectorMath;
import io.teknek.reckoning.math.VectorMathUtils;
import io.teknek.reckoning.math.WrappedForkJoinPool;
import io.teknek.reckoning.model.tensorparallel.StaticTensorParallelContext;
import io.teknek.reckoning.model.tensorparallel.SingleRankTensorParallelCollectives;
import io.teknek.reckoning.model.tensorparallel.GossipParallelMembership;
import io.teknek.reckoning.model.tensorparallel.TensorParallelCollectives;
import io.teknek.reckoning.model.tensorparallel.TensorParallelContext;
import io.teknek.reckoning.model.tensorparallel.TensorParallelPlanner;
import io.teknek.reckoning.safetensors.Config;
import io.teknek.reckoning.safetensors.LoraAdapter;
import io.teknek.reckoning.safetensors.LoraLayerDelta;
import io.teknek.reckoning.safetensors.ResolvedLoraAdapter;
import io.teknek.reckoning.safetensors.WeightLoader;
import io.teknek.reckoning.tensor2.TensorRef;
import io.teknek.reckoning.tensor2.TensorRefBackedTensor;
import io.teknek.reckoning.safetensors.fetch.LoraAdapterModelFetcher;
import io.teknek.reckoning.safetensors.prompt.PromptContext;
import io.teknek.reckoning.safetensors.prompt.PromptSupport;
import io.teknek.reckoning.tensor.*;
import io.teknek.reckoning.tensor.kv.KvBlock;
import io.teknek.reckoning.tensor.kv.KvBlockKey;
import io.teknek.reckoning.tensor.kv.KvBlockLease;
import io.teknek.reckoning.tensor.kv.KvBlockManager;
import io.teknek.reckoning.tensor.kv.KvCacheManager;
import io.teknek.reckoning.tensor.kv.KvCacheSession;
import io.teknek.reckoning.tensor.kv.KvPrefixSnapshotCache;
import io.teknek.reckoning.tensor.impl.Q8ByteBufferTensor;
import io.teknek.reckoning.tensor2.CompositeOps;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.MultiplyInPlace;
import io.teknek.reckoning.tensor2.ScaledSoftMax;
import io.teknek.reckoning.tensor2.TensorRef;
import io.teknek.reckoning.tensorlib.PlannedTensor;
import io.teknek.reckoning.tensorlib.PlannedTensorRef;
import io.teknek.reckoning.tensorlib.TensorPlan;
import io.teknek.reckoning.tensorlib.TensorPlanAdaptiveSplitTuner;
import io.teknek.reckoning.tensorlib.TensorRuntime;
import io.teknek.reckoning.tensorlib.TensorRuntimeMode;
import io.teknek.reckoning.toolcallparser.ToolCallParser;
import jdk.incubator.vector.FloatVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static io.teknek.reckoning.tensor.DebugSupport.debug;

/**
 * Base implementation for generation, classification, embedding, and shared model utilities.
 *
 * <h2>Prefix KV-cache contract</h2>
 * <p>Generation can reuse block-aligned KV prefixes through KVCache2. The cache is an internal
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
public abstract class AbstractModel implements Generator, Classifier, TensorPlanAdaptiveSplitTuner {
    static final Logger logger = LoggerFactory.getLogger(AbstractModel.class);

    public static final int DEFAULT_MAX_PREFILL_BATCH_SIZE = 512;
    private static final int[] ADAPTIVE_PREFILL_BATCH_SIZE_CANDIDATES = { 256, 512, 1024, 2048 };
    private static final long PREFILL_PROGRESS_INTERVAL_NANOS = TimeUnit.SECONDS.toNanos(3);
    private static final ThreadLocal<PrefillProgress> PREFILL_PROGRESS = new ThreadLocal<>();
    private static final ThreadLocal<Consumer<PrefillProgressEvent>> PREFILL_PROGRESS_LISTENER = new ThreadLocal<>();

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

    public record PrefillProgressEvent(int estimatedProcessedTokens, int totalTokens, int chunkStartPosition,
            int chunkEndPosition, int processedLayers, int totalLayers, double elapsedSeconds,
            double remainingSeconds, double tokensPerSecond) {
    }

    public static AutoCloseable withPrefillProgressListener(Consumer<PrefillProgressEvent> listener) {
        Consumer<PrefillProgressEvent> previous = PREFILL_PROGRESS_LISTENER.get();
        PREFILL_PROGRESS_LISTENER.set(Objects.requireNonNull(listener, "listener"));
        return () -> {
            if (previous == null) {
                PREFILL_PROGRESS_LISTENER.remove();
            } else {
                PREFILL_PROGRESS_LISTENER.set(previous);
            }
        };
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
            int tokensToProcessLength
    ) {
    }

    public record LayerDebugEvent(
            int layerIndex,
            String stage,
            TensorParallelContext tensorParallelContext,
            TensorRef hiddenStates
    ) {
    }

    /**
     * Forward execution boundary used by generation coordinators that do not own local KV memory directly.
     */
    public interface GenerationForwarder {
        TensorRef batchForward(int[] tokenIds, int startPosition);

        TensorRef forward(int tokenId, int position);
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
    protected SampleOutputRef sampleOutputRef;
    protected io.teknek.reckoning.generator2.TransformerBlock2[] transformerBlocks2;
    protected KvCacheManager kvCacheManager;
    protected KvPrefixSnapshotCache kvPrefixSnapshotCache;
    protected KvBlockManager kvBlockManager;
    protected final Lighter lighter;
    protected final CompositeOps compositeOps;
    protected final MetricRegistry metricRegistry;
    private final KvBufferCacheSettings kvBufferCacheSettings;
    protected final TensorParallelContext tensorParallelContext;
    protected final TensorParallelCollectives tensorParallelCollectives;
    private boolean gpuPrefillEnabled;
    private boolean gpuDecodeEnabled;
    private boolean gpuDecodeAttentionEnabled;
    private boolean gpuDiffusionBlockProjectionEnabled;
    private boolean packedBlockAttentionEnabled;
    private boolean packedPrefillEnabled = true;
    private boolean trackKvReadViewsEnabled;
    private GossipParallelMembership gossipParallelMembership;

    //embedding
    protected Optional<PoolingLayer> poolingLayer;

    protected final ToolCallParser toolCallParser;

    protected ClassifyOutput classifyOutput;
    protected WrappedForkJoinPool pool;
    protected PreTrainedTokenizer preTrainedTokenizer;
    protected int maxPrefillBatchSize = DEFAULT_MAX_PREFILL_BATCH_SIZE;
    protected TensorPlan modelLineagePlan;
    private final ConcurrentMap<String, TensorPlan.ImmutableTensor> modelLineageTensors = new ConcurrentHashMap<>();
    private final Queue<ModelLineageEntry> modelLineageEntries = new ConcurrentLinkedQueue<>();
    private final Queue<TensorRef> modelTensorRefs = new ConcurrentLinkedQueue<>();
    private final Set<TensorRef> registeredModelTensorRefSet = ConcurrentHashMap.newKeySet();
    private Optional<TensorRuntimeMode> tensorRuntimeMode = Optional.empty();
    private TensorRuntime tensorRuntime;
    private Map<String, Object> generationOptions = Map.of();
    private Optional<Integer> groupedDecodeQkvSplitSize = Optional.empty();
    private final ConcurrentMap<String, AdaptiveSplitState> adaptiveSplitStates = new ConcurrentHashMap<>();
    private final ConcurrentMap<String, AdaptiveAlternateState> adaptiveAlternateStates = new ConcurrentHashMap<>();
    private boolean initialized;
    private boolean tensorPlanTraceEnabled;
    private final ConcurrentMap<UUID, TensorPlanTraceContext> tensorPlanTraces = new ConcurrentHashMap<>();
    private final ThreadLocal<UUID> activeTensorPlanTraceId = new ThreadLocal<>();
    private volatile Consumer<GenerationDebugEvent> generationDebugHook = event -> {};
    private volatile Consumer<LayerDebugEvent> layerDebugHook = event -> {};
    private DeadReckoningTrace deadReckoningTrace;
    private String deadReckoningOwner;
    private String deadReckoningModel;

     protected AbstractModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer t, DType workingMemoryDType,
                              DType workingQType, Optional<DType> modelQType,
                               MetricRegistry metricRegistry, KvBufferCacheSettings kvBufferCacheSettings,
                             ToolCallParser toolCallParser, WrappedForkJoinPool pool,
                             TensorParallelContext tensorParallelContext,
                             TensorParallelCollectives tensorParallelCollectives,
                             Optional<DType> outputHeadQuantization, Lighter lighter) {
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
        this.metricRegistry = metricRegistry;
        this.lighter = Objects.requireNonNull(lighter, "lighter");
        this.lighter.setProviderObserver(event -> {
            String operation = event.operation();
            String providerName = event.kind().name().toLowerCase(java.util.Locale.ROOT);
            metricRegistry.counter("tensor2.provider." + operation + "." + providerName + ".attempted").inc();
            if (event.supported()) {
                metricRegistry.counter("tensor2.provider." + operation + "." + providerName + ".supported").inc();
                if (event.kind() == io.teknek.reckoning.tensor2.TensorProviderKind.NAIVE) {
                    metricRegistry.counter("tensor2.provider.naive.selected").inc();
                }
            } else {
                metricRegistry.counter("tensor2.provider." + operation + "." + providerName + ".unsupported").inc();
                metricRegistry.counter("tensor2.provider.fallback").inc();
            }
        });
        registerOptionalTensorRefNativeOps();
        w.setLighter(this.lighter);
        this.compositeOps = new CompositeOps(lighter, metricRegistry);
         this.kvBufferCacheSettings = kvBufferCacheSettings;
         this.kvBlockManager = new KvBlockManager(metricRegistry, kvBufferCacheSettings, lighter);
         this.kvCacheManager = new KvCacheManager(c.numberOfLayers, c.contextLength,
                 c.kvLength / tensorParallelContext.size(), workingMemoryDType, kvBufferCacheSettings,
                 metricRegistry, false, lighter);
         this.kvPrefixSnapshotCache = new KvPrefixSnapshotCache(c.numberOfLayers, c.contextLength,
                c.kvLength / tensorParallelContext.size(), kvBufferCacheSettings.getBlockSize(),
                kvBufferCacheSettings.getKvKeyDType(), kvBufferCacheSettings.getKvValueDType(),
                 lighter, metricRegistry, kvBufferCacheSettings);
        this.toolCallParser = toolCallParser;

         this.workingQType = resolveWorkingQType(workingQType);

        this.pool = pool;
        logger.debug("model constructor complete config={} inference_type={}", config.getClass().getSimpleName(), inferenceType);
    }

    public final void init() {
        if (initialized) {
            return;
        }
        this.modelLineagePlan = new TensorPlan(pool, metricRegistry, this, tensorRuntime, lighter);
        logger.debug("model init start config={} inference_type={}", config.getClass().getSimpleName(), inferenceType);
        this.embedInput = inferenceType.isInput ? loadInputWeights() : null;
        boolean tensorRefExecution = inferenceType.isFwdPass && usesTensorRefExecution();
        this.transformerBlocks2 = tensorRefExecution ? loadTransformerBlockWeights2() : null;
        this.sampleOutputRef = inferenceType.isOutput && tensorRefExecution ? loadOutputWeightsRef() : null;
        this.classifyOutput = inferenceType.isClassify ? loadClassifierWeights() : null;
        this.poolingLayer = inferenceType.isPooling ? Optional.ofNullable(loadPoolingWeights()) : Optional.empty();
        initialized = true;
        logger.debug("model init complete config={} inference_type={} layers={}", config.getClass().getSimpleName(),
                inferenceType, this.transformerBlocks2 == null ? 0 : this.transformerBlocks2.length);
    }


    private DType resolveWorkingQType(DType requestedWorkingQType) {
        if (requestedWorkingQType == null) {
            requestedWorkingQType = DType.I8;
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

        return requestedWorkingQType;
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

    void setTrackKvReadViewsEnabled(boolean trackKvReadViewsEnabled) {
        this.trackKvReadViewsEnabled = trackKvReadViewsEnabled;
        int[] configuredLayerKvLengths = kvCacheManager.layerKvLengths();
        this.kvCacheManager = new KvCacheManager(config.numberOfLayers, config.contextLength,
                config.kvLength / tensorParallelContext.size(), workingDType, kvBufferCacheSettings,
                metricRegistry, trackKvReadViewsEnabled, lighter);
        if (configuredLayerKvLengths != null) {
            this.kvCacheManager.setLayerKvLengths(configuredLayerKvLengths);
        }
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

    public boolean isTrackKvReadViewsEnabled() {
        return trackKvReadViewsEnabled;
    }

    public TensorParallelContext getTensorParallelContext() {
        return tensorParallelContext;
    }

    public TensorParallelCollectives getTensorParallelCollectives() {
        return tensorParallelCollectives;
    }

    
    public KvCacheManager kvCacheManager() {
        return kvCacheManager;
    }

    public KvCacheSession newKvCacheSession() {
        return kvCacheManager.openSession();
    }

    public KvPrefixSnapshotCache kvPrefixSnapshotCache() {
        return kvPrefixSnapshotCache;
    }

    public KvBufferCacheSettings.PrefixCacheMode prefixCacheMode() {
        return kvBufferCacheSettings.getPrefixCacheMode();
    }

    public KvBlockManager kvBlockManager() {
        return kvBlockManager;
    }

    public int restoreSharedPrefixToKvSession(int[] promptTokens, Optional<String> cacheSalt,
            KvCacheSession destination) {
        if (kvBufferCacheSettings.getPrefixCacheMode() != KvBufferCacheSettings.PrefixCacheMode.SHARED_BLOCKS) {
            return 0;
        }
        int restoredTokens = 0;
        for (KvBlockKey key : sharedPrefixBlockKeys(promptTokens, cacheSalt)) {
            KvBlockLease lease = kvBlockManager.retain(key, destination.sessionId());
            if (lease == null) {
                break;
            }
            destination.attachCommittedBlock(lease);
            restoredTokens += key.tokenCount();
            InferenceProfiler.counter(metricRegistry, "kvcache.v2.prefix.blocks.attached").inc();
            InferenceProfiler.counter(metricRegistry, "kvcache.v2.prefix.bytes.attached").inc(lease.block().encodedBytes());
        }
        if (restoredTokens > 0) {
            InferenceProfiler.counter(metricRegistry, "kvcache.v2.prefix.tokens.reused").inc(restoredTokens);
        }
        return restoredTokens;
    }

    public void storeSharedPrefixFromKvSession(int[] promptTokens, KvCacheSession source, Optional<String> cacheSalt) {
        if (kvBufferCacheSettings.getPrefixCacheMode() != KvBufferCacheSettings.PrefixCacheMode.SHARED_BLOCKS) {
            return;
        }
        for (KvBlockKey key : sharedPrefixBlockKeys(promptTokens, cacheSalt)) {
            if (keyEndPositionExclusive(key) > source.length() || source.committedBlockIsLeased(key.blockIndex())) {
                continue;
            }
            KvBlock block = source.detachCommittedBlock(key.blockIndex());
            KvBlockLease lease = kvBlockManager.admitAndRetain(key, block, source.sessionId(),
                    sharedPrefixDiskAdmitTokenCount(promptTokens));
            source.attachCommittedBlock(lease);
            InferenceProfiler.counter(metricRegistry, "kvcache.v2.prefix.blocks.admitted").inc();
        }
    }

    private int sharedPrefixDiskAdmitTokenCount(int[] promptTokens) {
        int blockSize = kvBufferCacheSettings.getBlockSize();
        int limit = Math.min(promptTokens.length, kvBufferCacheSettings.getMaxPrefixTokensPerPrompt());
        return (limit / blockSize) * blockSize;
    }

    private List<KvBlockKey> sharedPrefixBlockKeys(int[] promptTokens, Optional<String> cacheSalt) {
        int blockSize = kvBufferCacheSettings.getBlockSize();
        int limit = Math.min(promptTokens.length, kvBufferCacheSettings.getMaxPrefixTokensPerPrompt());
        int fullTokens = (limit / blockSize) * blockSize;
        if (fullTokens == 0) {
            return List.of();
        }
        ArrayList<KvBlockKey> keys = new ArrayList<>(fullTokens / blockSize);
        long parentHash = 0L;
        for (int blockIndex = 0; blockIndex < fullTokens / blockSize; blockIndex++) {
            long tokenBlockHash = tokenBlockHash(parentHash, promptTokens, blockIndex * blockSize, blockSize);
            keys.add(new KvBlockKey(1, effectiveModelFingerprint(), adapterFingerprint(), tokenizerFingerprint(),
                    cacheSalt.orElse(""), ropeConfigHash(), attentionConfigHash(), blockIndex, parentHash,
                    tokenBlockHash, blockSize, blockSize, config.numberOfLayers, getLocalKvLength(),
                    kvBufferCacheSettings.getKvKeyDType(), kvBufferCacheSettings.getKvValueDType(),
                    kvBufferCacheSettings.getKvBlockStoragePolicy() == KvBufferCacheSettings.KvBlockStoragePolicy.DENSE
                            ? io.teknek.reckoning.tensor.kv.KvBlockLayout.DENSE
                            : io.teknek.reckoning.tensor.kv.KvBlockLayout.MSE_TURBOQUANT,
                    kvBufferCacheSettings.getKvTurboQuantBits(), tensorParallelContext.size(),
                    tensorParallelContext.rank(), 0L, localShardId()));
            parentHash = tokenBlockHash;
        }
        return keys;
    }

    private int keyEndPositionExclusive(KvBlockKey key) {
        return key.blockIndex() * key.blockSize() + key.tokenCount();
    }

    private String effectiveModelFingerprint() {
        return stableHash("model", weights.getClass().getName(), weights.modelRoot().map(Object::toString).orElse("none"),
                weights.getModelDType().name(), config.getClass().getName(), String.valueOf(config.contextLength),
                String.valueOf(config.embeddingLength), String.valueOf(config.hiddenLength),
                String.valueOf(config.numberOfHeads), String.valueOf(config.numberOfKeyValueHeads),
                String.valueOf(config.headSize), String.valueOf(config.numberOfLayers), String.valueOf(config.vocabularySize),
                String.valueOf(config.bosToken), String.valueOf(config.eosTokens), String.valueOf(config.architectures));
    }

    private String adapterFingerprint() {
        return activeLoraAdapterId().map(id -> stableHash("active-lora", id)).orElse("none");
    }

    private String tokenizerFingerprint() {
        return stableHash("tokenizer", tokenizer == null ? "none" : tokenizer.getClass().getName(),
                preTrainedTokenizer == null ? "none" : preTrainedTokenizer.getClass().getName());
    }

    private String ropeConfigHash() {
        String sample = config.ropeFreqs.map(freqs -> freqs.length == 0 || freqs[0].length == 0
                ? "empty"
                : freqs[0][0] + ":" + freqs[Math.min(freqs.length - 1, 1)][Math.min(freqs[0].length - 1, 1)])
                .orElse("none");
        return stableHash("rope", String.valueOf(config.ropeFreqs.isPresent()), String.valueOf(config.contextLength),
                String.valueOf(config.headSize), sample);
    }

    private String attentionConfigHash() {
        return stableHash("attention", String.valueOf(config.attentionLength), String.valueOf(config.headGroupSize),
                String.valueOf(config.isGQA), String.valueOf(config.attnLogitSoftCapping),
                String.valueOf(config.residualMultiplier), String.valueOf(config.attentionMultiplier));
    }

    private String localShardId() {
        return "tpSize=" + tensorParallelContext.size() + ":tpRank=" + tensorParallelContext.rank()
                + ":localKvLength=" + getLocalKvLength();
    }

    private static long tokenBlockHash(long parentHash, int[] tokens, int offset, int length) {
        long hash = 0xcbf29ce484222325L ^ parentHash;
        for (int i = 0; i < length; i++) {
            hash ^= tokens[offset + i];
            hash *= 0x100000001b3L;
        }
        return hash;
    }

    private static String stableHash(String prefix, String... parts) {
        long hash = 0xcbf29ce484222325L;
        for (String part : parts) {
            String value = part == null ? "null" : part;
            for (int i = 0; i < value.length(); i++) {
                hash ^= value.charAt(i);
                hash *= 0x100000001b3L;
            }
            hash ^= 0xff;
            hash *= 0x100000001b3L;
        }
        return prefix + ":" + Long.toUnsignedString(hash, 16);
    }

    public boolean usesKvCache2Generation() {
        return false;
    }

    public void emitPromptPrefillDebug(int[] promptTokens, int prefixLength, int startPosition,
            int tokensToProcessLength) {
        generationDebugHook.accept(new GenerationDebugEvent(
                GenerationDebugEventType.AFTER_PROMPT_PREFILL,
                promptTokens,
                prefixLength,
                startPosition,
                tokensToProcessLength));
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

    void configureDeadReckoning(DeadReckoningParams params, String owner, String model) {
        if (deadReckoningTrace != null) {
            deadReckoningTrace.close();
            deadReckoningTrace = null;
        }
        if (params != null) {
            deadReckoningTrace = new DeadReckoningTrace(params);
            deadReckoningOwner = owner;
            deadReckoningModel = model;
            setLayerDebugHook(deadReckoningTrace::recordLayer);
        }
    }

    void deadReckoningHeader(PromptContext prompt, GeneratorParameters parameters) {
        if (deadReckoningTrace != null) {
            deadReckoningTrace.recordHeader(deadReckoningOwner, deadReckoningModel, prompt, parameters,
                    Arrays.stream(constructPromptTokens(encodeText(prompt.getPrompt())))
                            .asLongStream().toArray());
        }
    }

    void deadReckoningResponse(Response response) {
        if (deadReckoningTrace != null) {
            deadReckoningTrace.recordResponse(response);
        }
    }

    public void emitLayerDebug(int layerIndex, String stage, AbstractTensor hiddenStates) {
        try (TensorRef tensor = TensorRef.borrowed(hiddenStates)) {
            emitLayerDebug(layerIndex, stage, tensor);
        }
    }

    public void emitLayerDebug(int layerIndex, String stage, TensorRef hiddenStates) {
        layerDebugHook.accept(new LayerDebugEvent(layerIndex, stage, tensorParallelContext, hiddenStates));
    }

    void emitGenerationDebug(GenerationDebugEvent event) {
        generationDebugHook.accept(event);
        if (deadReckoningTrace != null) {
            deadReckoningTrace.recordGenerationDebug(event);
        }
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
        adapter.setLighter(getLighter());
        registeredLoraAdapters.put(adapterId, new ResolvedLoraAdapter(adapter, getWorkingDType(), getLighter()));
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
     * io.teknek.reckoning.model.LocalGenerationBackend}) -- KV state computed under one adapter
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
    protected GenerationStepInputs prepareInputsForGeneration(int[] inputIds, Integer nextSequenceLength, PastKeyValues pastKeyValues, int[] attentionMask, int[] encoderAttentionMask, TensorRef inputsEmbeds, boolean firstIteration, int[] positionIds, int[] tokenTypeIds, int[] mmTokenTypeIds) {
        return GenerationInputPreparer.prepareInputsForGeneration(config, inputIds, nextSequenceLength, pastKeyValues, attentionMask, encoderAttentionMask, inputsEmbeds, firstIteration, positionIds, tokenTypeIds, mmTokenTypeIds,
                this::makeDenseTensorRef, lighter);
    }

    protected abstract EmbedInput loadInputWeights();
    /** Loads output-head tensors directly in the TensorRef representation. */
    protected SampleOutputRef loadOutputWeightsRef() {
        throw new UnsupportedOperationException("TensorRef output weights are not implemented for "
                + getClass().getName());
    }
    /** Optional TensorRef-native transformer block path. */
    protected io.teknek.reckoning.generator2.TransformerBlock2[] loadTransformerBlockWeights2() {
        return null;
    }

    protected boolean usesTensorRefExecution() {
        return false;
    }

    private void registerOptionalTensorRefNativeOps() {
        try {
            Class<?> nativeOpsClass = Class.forName("io.teknek.reckoning.tensor2.NativeOps");
            boolean available = (Boolean) nativeOpsClass.getMethod("isAvailable").invoke(null);
            if (available) {
                Object nativeOps = nativeOpsClass.getConstructor().newInstance();
                lighter.putTensorOperations(io.teknek.reckoning.tensor2.TensorProviderKind.SIMD,
                        (io.teknek.reckoning.tensor2.TensorOps) nativeOps);
                logger.info("TensorRef native operations enabled");
            } else {
                logger.warn("TensorRef native operations unavailable; using Panama fallback");
            }
        } catch (ClassNotFoundException ignored) {
            // Native Tensor2 operations are optional for core/runtime deployments.
        } catch (ReflectiveOperationException | RuntimeException failure) {
            logger.warn("TensorRef native operation discovery failed; using Panama fallback: {}",
                    failure.toString());
        }
    }

    @Override
    public void close() {
        if (deadReckoningTrace != null) {
            deadReckoningTrace.close();
            deadReckoningTrace = null;
        }
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
        if (kvPrefixSnapshotCache != null) {
            kvPrefixSnapshotCache.close();
        }
        if (kvBlockManager != null) {
            kvBlockManager.close();
        }
        modelTensorRefs.forEach(TensorRef::close);
        modelTensorRefs.clear();
        registeredModelTensorRefSet.clear();
        try {
            weights.close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public Optional<GossipParallelMembership> gossipParallelMembership() {
        return Optional.ofNullable(gossipParallelMembership);
    }

    void setMaxPrefillBatchSize(int maxPrefillBatchSize) {
        if (maxPrefillBatchSize < 1) {
            throw new IllegalArgumentException("maxPrefillBatchSize must be >= 1");
        }
        this.maxPrefillBatchSize = maxPrefillBatchSize;
    }

    void setGossipParallelMembership(GossipParallelMembership gossipParallelMembership) {
        this.gossipParallelMembership = Objects.requireNonNull(gossipParallelMembership, "gossipParallelMembership");
    }

    public Config getConfig(){
        return config;
    }

    /** Allocates a model-working-dtype tensor2 result owned by the caller. */
    public TensorRef makeTensorRef(int... shape) {
        return lighter.allocate(workingDType, TensorShape.of(shape));
    }

    /** Allocates a dense model-working-dtype tensor2 result owned by the caller. */
    public TensorRef makeDenseTensorRef(int... shape) {
        return lighter.allocate(workingDType, TensorShape.of(shape));
    }

    /** Allocates a dense model-working-dtype tensor2 result owned by the caller. */
    public TensorRef makeDenseTensorRef(TensorShape s) {
        return lighter.allocate(workingDType, s);
    }

    /** Temporary adapter for model components that still consume the legacy tensor type. */
    protected AbstractTensor loadLegacyWeight(String name) {
        TensorRef ref = weights.load(name);
        try {
            return TensorRefBackedTensor.as(ref, ref.dType());
        } catch (RuntimeException e) {
            ref.close();
            throw e;
        }
    }

    /** Temporary adapter for legacy row-sliced model components. */
    protected AbstractTensor loadLegacyWeightRows(String name, int rowOffset, int rowCount) {
        TensorRef ref = weights.loadRows(name, rowOffset, rowCount);
        try {
            return TensorRefBackedTensor.as(ref, ref.dType());
        } catch (RuntimeException e) {
            ref.close();
            throw e;
        }
    }

    public DType getWorkingDType() {
        return workingDType;
    }

    public DType getWorkingQType() {
        return workingQType;
    }

    public DType getModelDType() {
        return modelDType;
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
        return tokenizer.decode(new io.teknek.reckoning.grace.TokenIds(Ints.checkedCast(token)), false, false, false, false);
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
        if (addBosToken() && (encoded.length == 0 || encoded[0] != config.bosToken)) {
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

    SamplerReturn createNextTokenRef(GeneratorParameters parameters, TensorRef output, TensorRef logits,
            ResponseContext responseContext, Random random, Optional<LogitsProcessorRef> processor,
            TensorRef argMaxScratch) {
        if (sampleOutputRef == null) {
            throw new IllegalStateException("TensorRef output weights are not loaded");
        }
        return new ReckoningSamplerRef(this, parameters, output, logits, sampleOutputRef.outputLayerNorm(),
                sampleOutputRef.outputLogitsWeights(), random, responseContext, argMaxScratch, processor).sample();
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
        deadReckoningHeader(promptContext, generatorParameters);
        Response response = new GenerationEngineRef().generate(this, new ForwarderGenerationBackend(forwarder), sessionId,
                promptContext, generatorParameters, eventFired);
        deadReckoningResponse(response);
        return response;
    }

    /** Runs standard generation through the TensorRef-native backend boundary. */
    public Response generateWithBackendRef(UUID sessionId, PromptContext promptContext,
            GeneratorParameters generatorParameters, GenerateEvent eventFired, GenerationBackendRef backend) {
        Objects.requireNonNull(sessionId, "sessionId");
        Objects.requireNonNull(backend, "backend");
        deadReckoningHeader(promptContext, generatorParameters);
        Response response = new GenerationEngineRef().generate(this, backend, sessionId, promptContext, generatorParameters,
                eventFired);
        deadReckoningResponse(response);
        return response;
    }

    @Override
    public SortedMap<String, Float> classify(String input, PoolingType poolingType) {
        if (!config.isClassifier()) {
            throw new UnsupportedOperationException("Classification not supported by this model");
        }
        deadReckoningTaskStart("CLASSIFICATION", input, poolingType.name());
        SortedMap<String, Float> result = classifyInput(input, poolingType);
        deadReckoningClassificationResult(result);
        return result;
    }

    /**
     * Runs the model-specific classification path for one input.
     *
     * <p>The default path preserves the existing BERT-style behavior by embedding the
     * input first and applying a single projection. Encoder families whose classifier
     * consumes unpooled sequence output can override this method.</p>
     */
    protected SortedMap<String, Float> classifyInput(String input, PoolingType poolingType) {
        float[] embedding = embed(input, poolingType);
        return classifyEmbedding(embedding);
    }

    /**
     * Applies the model's classification head to one encoded representation.
     *
     * <p>The default implementation preserves the original single linear projection
     * behavior. Model families with a multi-stage head can override this seam without
     * changing the public {@link #classify(String, PoolingType)} API.</p>
     */
    protected SortedMap<String, Float> classifyEmbedding(float[] embedding) {
        try (TensorRef input = makeDenseTensorRef(1, embedding.length)) {
            for (int column = 0; column < embedding.length; column++) {
                input.set(embedding[column], 0, column);
            }
            return classifyTensor(input);
        }
    }

    /** Applies a single linear classifier using only the TensorRef execution stack. */
    protected SortedMap<String, Float> classifyTensor(TensorRef input) {
        TensorRef weights = classifyOutput.getClassificationWeights();
        int classes = (int) weights.shape().first();
        try (TensorRef scores = makeDenseTensorRef(1, classes)) {
            metricRegistry.timer("classify.1_dotproduct_scores").time(() ->
                    lighter.dotProductRows(scores, input, weights, 0, config.embeddingLength,
                            0, classes, 0));
            metricRegistry.timer("classify.2_accumulate_scores_bias").time(() ->
                    classifyOutput.getClassificationBias().ifPresent(bias -> {
                        for (int label = 0; label < classes; label++) {
                            scores.set(scores.get(0, label) + bias.get(0, label), 0, label);
                        }
                    }));
            metricRegistry.timer("classify.3_softmax_scores").time(() ->
                    compositeOps.scaledSoftMax(new ScaledSoftMax(1.0f).target(scores).offsetAndLength(0, classes)));
            SortedMap<String, Float> result = new TreeMap<>();
            for (int i = 0; i < classes; i++) {
                result.put(config.classifcationLabels.get().inverse().get(i), scores.get(0, i));
            }
            return result;
        }
    }

    public float[] embed(String input, PoolingType poolingType) {
        deadReckoningTaskStart("EMBEDDING", input, poolingType.name());
        //TODO better recipe then this? timed callable
        Timer.Context c = metricRegistry.timer("abstractmodel.embed").time();
        try {
            float[] result = timedEmbedding(input, poolingType);
            deadReckoningEmbeddingResult(result);
            return result;
        } finally {
            c.stop();
        }
    }

    private void deadReckoningTaskStart(String task, String input, String poolingType) {
        if (deadReckoningTrace != null) {
            deadReckoningTrace.recordTaskStart(task, input, poolingType);
        }
    }

    private void deadReckoningEmbeddingResult(float[] embedding) {
        if (deadReckoningTrace != null) {
            deadReckoningTrace.recordEmbeddingResult(embedding);
        }
    }

    private void deadReckoningClassificationResult(SortedMap<String, Float> scores) {
        if (deadReckoningTrace != null) {
            deadReckoningTrace.recordClassificationResult(scores);
        }
    }
    protected float[] timedEmbedding(String input, PoolingType poolingType) {
        CausualWhisperer.LOGGER.debug("embedding on {} using pooling type {}", input, poolingType);
        int[] encoded = Arrays.stream(encodeText(input)).mapToInt(Ints::checkedCast).toArray();
        Preconditions.checkArgument(encoded.length < config.contextLength);
        float [] outputEmbedding = new float[config.embeddingLength];
        CausualWhisperer.LOGGER.debug("created float [] outputEmbedding of length {}", config.embeddingLength);

        try (LocalGenerationBackend backend = new LocalGenerationBackend(this);
             GenerationBackendRef.GenerationSessionRef session = backend.openRef(java.util.UUID.randomUUID(), encoded,
                     new GeneratorParameters());
             TensorRef r = session.prefill(GenerationCursor.from(encoded, session.prefixLength()))) {
            int promptLength = encoded.length;
            float avgp = 1.0f / promptLength;
            CausualWhisperer.LOGGER.debug("1.0f / promptLength {} = avgp {}", promptLength, avgp);

            try (Timer.Context ignored = metricRegistry.timer("abstractmodel.embed_1_batchforward").time()) {
                if (poolingType == PoolingType.MODEL){
                    if (poolingLayer.isEmpty()){
                        throw new UnsupportedOperationException("no pooling layer for this model");
                    }
                    try (TensorRef output = r.slice(promptLength - 1);
                         TensorRef pooled = makeDenseTensorRef(1, config.embeddingLength)) {
                        lighter.batchDotProduct(new io.teknek.reckoning.tensor2.BatchDotProduct()
                                .result(pooled).a(output).b(poolingLayer.get().getPoolingWeights())
                                .aColumnOffset(0).bColumnOffset(0).columnLength(config.embeddingLength)
                                .resultRowOffset(0).bRowOffset(0).rowChunkSize(config.embeddingLength));
                        poolingLayer.get().getPoolingBias().ifPresent(bias ->
                                lighter.accumulate(new io.teknek.reckoning.tensor2.Accumulate(bias)
                                        .into(pooled).offsetAndLength(0, config.embeddingLength)));
                        VectorMath.pfor(0, config.embeddingLength, i -> {
                            outputEmbedding[i] = ActivationFunction.eval(config.activationFunction, pooled.get(0, i));
                        }, pool);
                        return outputEmbedding;
                    }
                }
                for (int i = 0; i < promptLength; i++) {
                    try (TensorRef output = r.slice(i)) {
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
                }
                VectorMathUtils.l2normalize(outputEmbedding);
                return outputEmbedding;
            }
        }
    }


    /** Direct TensorRef generation entrypoint for the KV-cache v2 path. */
    public TensorRef batchForwardRef(int[] tokenIds, int startPos, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer) {
        Preconditions.checkArgument(tokenIds.length > 0, "tokenIds must not be empty");
        TensorRef lastBatchOutput = null;
        for (int i = 0; i < tokenIds.length; i += maxPrefillBatchSize) {
            int[] batch = Arrays.copyOfRange(tokenIds, i, Math.min(tokenIds.length, i + maxPrefillBatchSize));
            TensorRef input = embedInput.batchInputsToEmbeddings(batch, startPos + i);
            PlannedTensorRef planned = plannedInputEmbeddingsRef("input_embeddings", input,
                    io.teknek.reckoning.generator.ForwardPhase.PREFILL);
            PlannedTensorRef result = forward(planned, startPos + i, kvSession, tensorReducer,
                    io.teknek.reckoning.generator.ForwardPhase.PREFILL);
            if (lastBatchOutput != null) {
                lastBatchOutput.close();
            }
            lastBatchOutput = result.tensor();
            kvSession.advanceLength(startPos + i + batch.length);
        }
        return lastBatchOutput;
    }

    public TensorRef batchForwardRef(int[] tokenIds, int startPos, KvCacheSession kvSession) {
        return batchForwardRef(tokenIds, startPos, kvSession, Optional.empty());
    }

    /** Direct TensorRef decode entrypoint for the KV-cache v2 path. */
    public TensorRef forwardRef(int tokenId, int pos, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer) {
        TensorRef embedding = embedInput.inputTokenToEmbedding(tokenId, pos);
        PlannedTensorRef planned = plannedInputEmbeddingsRef("input_embedding", embedding,
                io.teknek.reckoning.generator.ForwardPhase.DECODE);
        PlannedTensorRef result = forward(planned, pos, kvSession, tensorReducer,
                io.teknek.reckoning.generator.ForwardPhase.DECODE);
        kvSession.advanceLength(pos + 1);
        return result.tensor();
    }

    public TensorRef forwardRef(int tokenId, int pos, KvCacheSession kvSession) {
        return forwardRef(tokenId, pos, kvSession, Optional.empty());
    }


    protected PlannedTensorRef plannedInputEmbeddingsRef(String name, TensorRef embeddings,
            io.teknek.reckoning.generator.ForwardPhase phase) {
        TensorPlan.Tensor lineage = modelLineageTensor("model.embed_tokens.weight")
                .map(upstream -> modelLineagePlan.input(name, upstream, embeddings))
                .orElseGet(() -> modelLineagePlan.input(name, embeddings))
                .as(name);
        traceTensorPlan(getClass().getSimpleName(), "embedinput." + name, phase.name(), -1, "N/A", lineage.plan());
        return new PlannedTensorRef(embeddings, lineage);
    }

    public PlannedTensorRef forward(PlannedTensorRef embedding, int startPos, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, io.teknek.reckoning.generator.ForwardPhase phase) {
        return forward(embedding, startPos, kvSession, tensorReducer, phase,
                1, (int) embedding.tensor().shape().first(), null);
    }

    public PlannedTensorRef forward(PlannedTensorRef embedding, int startPos, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, io.teknek.reckoning.generator.ForwardPhase phase,
            int batchSize, int sequenceLength, int[] attentionMask) {
        Preconditions.checkState(transformerBlocks2 != null, "TensorRef transformer blocks are not initialized");
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, "abstractmodel.forward_layers").time()) {
            emitLayerDebug(-1, "input", embedding.tensor());
            for (int layer = 0; layer < config.numberOfLayers; layer++) {
                throwIfGenerationInterrupted();
                TensorRef input = embedding.tensor();
                TensorRef output = transformerBlocks2[layer].forward(input, startPos, kvSession, tensorReducer, phase,
                        batchSize, sequenceLength, attentionMask);
                embedding = new PlannedTensorRef(output, embedding.plan());
                input.close();
                emitLayerDebug(layer, "layer_output", output);
            }
            return embedding;
        }
    }

    public PlannedTensor forward(PlannedTensor embedding, int startPos, KvCacheSession kvSession,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer, io.teknek.reckoning.generator.ForwardPhase phase) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, "abstractmodel.forward_layers").time()) {
            emitLayerDebug(-1, "input", embedding.tensor());
            int batchTokens = embedding.tensor().shape().first();
            for (int i = 0; i < config.numberOfLayers; i++) {
                throwIfGenerationInterrupted();
                int relativeLayer = i;
                AbstractTensor ref = embedding.tensor();
                TensorRef inputRef = TensorRef.borrowed(embedding.tensor());
                TensorRef outputRef = transformerBlocks2[relativeLayer].forward(inputRef, startPos, kvSession,
                        Optional.empty(), phase);
                embedding = new PlannedTensor(new TensorRefBackedTensor(outputRef), embedding.plan());
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
        PrefillProgressEvent event = new PrefillProgressEvent(estimatedProcessedTokens, progress.totalTokens,
                chunkStartPosition, chunkEndPosition, processedLayers, totalLayers, elapsedSeconds, remainingSeconds,
                tokensPerSecond);
        Consumer<PrefillProgressEvent> listener = PREFILL_PROGRESS_LISTENER.get();
        if (listener != null) {
            listener.accept(event);
        }
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

    /** Close-safe TensorRef equivalent of {@link #maybeQuantizeReadOnly(AbstractTensor, String)}. */
    public TensorRefLease maybeQuantizeReadOnly(TensorRef t, String counterPrefix) {
        if (t.dType() == workingQType) {
            InferenceProfiler.counter(metricRegistry, counterPrefix + ".read_only").inc();
            return new TensorRefLease(t, false);
        }
        InferenceProfiler.counter(metricRegistry, counterPrefix + ".copy_or_quantize").inc();
        return new TensorRefLease(lighter.reshape(t, workingQType), true);
    }

    public record TensorRefLease(TensorRef tensor, boolean owned) implements AutoCloseable {
        @Override
        public void close() {
            if (owned) {
                tensor.close();
            }
        }
    }

    public PreTrainedTokenizer getTokenizer(){
        return this.tokenizer;
    }

    public boolean isSpecialToken(int token) {
        return tokenizer.allSpecialIds().contains(token);
    }

    public AbstractTensor allocateDirty(DType dtype, TensorShape shape) {
        return new TensorRefBackedTensor(lighter.allocateDirty(dtype, shape));
    }

    protected  ClassifyOutput loadClassifierWeights(){
        return null;
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

    public Lighter getLighter() {
        return lighter;
    }

    public CompositeOps getCompositeOps() {
        return compositeOps;
    }

    public void scale(float factor, AbstractTensor target, int offset, int length) {
        try (TensorRef targetRef = TensorRef.borrowed(target)) {
            compositeOps.multiplyInPlace(new MultiplyInPlace(factor).target(targetRef).offsetAndLength(offset, length));
        }
    }

    /**
     * Runs chunk work on the model-owned executor.
     *
     * <p>Operations invoked by {@code action} must not use provider-managed threading
     * ({@code Threading.PROVIDER}). The callback already runs on model executor workers;
     * blocking those workers while a nested provider pool executes can exhaust the
     * bounded pools and deadlock the request. Use caller-managed/synchronous provider
     * execution for operations inside this callback.</p>
     */
    public void runChunks(String operation, int offset, int length, int splitSize, Optional<AbstractTensor> localityTensor,
            io.teknek.reckoning.math.BiIntConsumer action) {
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

    public Optional<Integer> groupedDecodeQkvSplitSize() {
        return groupedDecodeQkvSplitSize;
    }

    void setGroupedDecodeQkvSplitSize(Optional<Integer> groupedDecodeQkvSplitSize) {
        this.groupedDecodeQkvSplitSize = Objects.requireNonNull(groupedDecodeQkvSplitSize, "groupedDecodeQkvSplitSize");
    }

    @Override
    public int chooseSplit(String planName, TensorShape shape, int defaultSplit, int minSplit, int maxSplit) {
        String key = adaptiveSplitKey(planName, shape);
        AdaptiveSplitState state = adaptiveSplitStates.computeIfAbsent(key,
                ignored -> new AdaptiveSplitState(minSplit, maxSplit, defaultSplit));
        int split = state.choose();
        if (InferenceProfiler.isEnabled()) {
            InferenceProfiler.counter(metricRegistry, "tensorplan.dit.split", "plan", planName, "split",
                    Integer.toString(split)).inc();
        }
        return split;
    }

    @Override
    public void observeSplit(String planName, TensorShape shape, int split, long elapsedNanos) {
        AdaptiveSplitState state = adaptiveSplitStates.get(adaptiveSplitKey(planName, shape));
        if (state != null) {
            boolean promoted = state.observe(split, elapsedNanos);
            if (InferenceProfiler.isEnabled()) {
                InferenceProfiler.timer(metricRegistry, "tensorplan.dit.elapsed", "plan", planName, "split",
                        Integer.toString(split)).update(elapsedNanos, TimeUnit.NANOSECONDS);
                if (promoted) {
                    InferenceProfiler.counter(metricRegistry, "tensorplan.dit.selected", "plan", planName, "split",
                            Integer.toString(state.selected())).inc();
                }
            }
        }
    }

    @Override
    public String chooseAlternate(String planName, List<String> candidates) {
        AdaptiveAlternateState state = adaptiveAlternateStates.computeIfAbsent(planName,
                ignored -> new AdaptiveAlternateState(candidates));
        String candidate = state.choose(candidates);
        if (InferenceProfiler.isEnabled()) {
            InferenceProfiler.counter(metricRegistry, "tensorplan.dit.alternate.choice", "plan", planName,
                    "candidate", candidate).inc();
        }
        return candidate;
    }

    @Override
    public void observeAlternate(String planName, String candidate, long elapsedNanos) {
        observeAlternate(planName, candidate, elapsedNanos, Map.of());
    }

    @Override
    public void observeAlternate(String planName, String candidate, long elapsedNanos, Map<String, String> tags) {
        AdaptiveAlternateState state = adaptiveAlternateStates.get(planName);
        if (state != null) {
            boolean promoted = state.observe(candidate, elapsedNanos);
            if (InferenceProfiler.isEnabled()) {
                InferenceProfiler.timer(metricRegistry, "tensorplan.dit.alternate.elapsed",
                        profilerTags(planName, candidate, tags)).update(elapsedNanos, TimeUnit.NANOSECONDS);
                if (promoted) {
                    InferenceProfiler.counter(metricRegistry, "tensorplan.dit.alternate.selected", "plan", planName,
                            "candidate", state.selected()).inc();
                }
            }
        }
    }

    private static String[] profilerTags(String planName, String candidate, Map<String, String> tags) {
        List<String> entries = new ArrayList<>(2 + tags.size() * 2);
        entries.add("plan");
        entries.add(planName);
        entries.add("candidate");
        entries.add(candidate);
        tags.entrySet().stream().sorted(Map.Entry.comparingByKey()).forEach(entry -> {
            entries.add(entry.getKey());
            entries.add(entry.getValue());
        });
        return entries.toArray(String[]::new);
    }

    private static String adaptiveSplitKey(String planName, TensorShape shape) {
        return planName + "|" + shape;
    }

    private static final class AdaptiveSplitState {
        private static final double EPSILON = 0.05;
        private static final double ALPHA = 0.15;
        private final int[] splits;
        private final long[] counts;
        private final double[] ewmaNanos;
        private final Random random = new Random(0x51A7E5EEDL);
        private final int defaultIndex;
        private int lastBestIndex;

        private AdaptiveSplitState(int minSplit, int maxSplit, int defaultSplit) {
            this.splits = new int[maxSplit - minSplit + 1];
            for (int i = 0; i < splits.length; i++) {
                splits[i] = minSplit + i;
            }
            this.counts = new long[splits.length];
            this.ewmaNanos = new double[splits.length];
            this.defaultIndex = Math.max(0, Math.min(splits.length - 1, defaultSplit - minSplit));
            this.lastBestIndex = defaultIndex;
        }

        synchronized int choose() {
            int unsampled = firstUnsampledIndex();
            if (unsampled >= 0) {
                return splits[unsampled];
            }
            if (random.nextDouble() < EPSILON) {
                return splits[random.nextInt(splits.length)];
            }
            lastBestIndex = bestIndex();
            return splits[lastBestIndex];
        }

        synchronized boolean observe(int split, long elapsedNanos) {
            int previousBest = bestIndex();
            int index = indexOf(split);
            if (index < 0) {
                return false;
            }
            counts[index]++;
            ewmaNanos[index] = counts[index] == 1
                    ? elapsedNanos
                    : (ALPHA * elapsedNanos) + ((1.0 - ALPHA) * ewmaNanos[index]);
            lastBestIndex = bestIndex();
            return lastBestIndex != previousBest;
        }

        synchronized int selected() {
            return splits[bestIndex()];
        }

        private int firstUnsampledIndex() {
            for (int offset = 0; offset < splits.length; offset++) {
                int index = (defaultIndex + offset) % splits.length;
                if (counts[index] == 0) {
                    return index;
                }
            }
            return -1;
        }

        private int bestIndex() {
            int bestIndex = lastBestIndex;
            double bestMean = Double.POSITIVE_INFINITY;
            for (int i = 0; i < splits.length; i++) {
                if (counts[i] == 0) {
                    continue;
                }
                if (ewmaNanos[i] < bestMean) {
                    bestMean = ewmaNanos[i];
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        private int indexOf(int split) {
            for (int i = 0; i < splits.length; i++) {
                if (splits[i] == split) {
                    return i;
                }
            }
            return -1;
        }
    }

    private static final class AdaptiveAlternateState {
        private static final double EPSILON = 0.05;
        private static final double ALPHA = 0.15;
        private final Random random = new Random(0xA17E5EEDL);
        private final Map<String, Stats> stats = new LinkedHashMap<>();
        private String lastBest;

        private AdaptiveAlternateState(List<String> candidates) {
            for (String candidate : candidates) {
                stats.put(candidate, new Stats());
            }
            lastBest = candidates.isEmpty() ? null : candidates.getFirst();
        }

        synchronized String choose(List<String> candidates) {
            for (String candidate : candidates) {
                stats.computeIfAbsent(candidate, ignored -> new Stats());
                if (stats.get(candidate).count == 0) {
                    return candidate;
                }
            }
            if (random.nextDouble() < EPSILON) {
                return candidates.get(random.nextInt(candidates.size()));
            }
            lastBest = best(candidates);
            return lastBest;
        }

        synchronized boolean observe(String candidate, long elapsedNanos) {
            String previousBest = best(List.copyOf(stats.keySet()));
            Stats stat = stats.computeIfAbsent(candidate, ignored -> new Stats());
            stat.count++;
            stat.ewmaNanos = stat.count == 1
                    ? elapsedNanos
                    : (ALPHA * elapsedNanos) + ((1.0 - ALPHA) * stat.ewmaNanos);
            lastBest = best(List.copyOf(stats.keySet()));
            return !Objects.equals(previousBest, lastBest);
        }

        synchronized String selected() {
            return lastBest;
        }

        private String best(List<String> candidates) {
            String best = lastBest == null ? candidates.getFirst() : lastBest;
            double bestMean = Double.POSITIVE_INFINITY;
            for (String candidate : candidates) {
                Stats stat = stats.get(candidate);
                if (stat == null || stat.count == 0) {
                    continue;
                }
                if (stat.ewmaNanos < bestMean) {
                    bestMean = stat.ewmaNanos;
                    best = candidate;
                }
            }
            return best;
        }

        private static final class Stats {
            private long count;
            private double ewmaNanos;
        }
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

    protected TensorRef registerModelTensorRef(TensorRef ref) {
        TensorRef registered = Objects.requireNonNull(ref, "ref");
        if (registeredModelTensorRefSet.add(registered)) {
            modelTensorRefs.add(registered);
        }
        return registered;
    }

    protected TensorRef loadAndMaybeQuantizedExcluding1DTensors(String name,
            Function<String, TensorRef> loader, DType qType) {
        TensorRef ref = loader.apply(name);
        if (ref.shape().first() == 1) {
            return ref;
        }
        if (lighter.shouldQuantizeForEfficiency(ref, qType)) {
            TensorRef quantized = lighter.reshape(ref, qType);
            ref.close();
            return quantized;
        }
        return ref;
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
                + "tensorProviders=" + lighter.tensorOperations().keySet() + '\n'
                + "tensorRuntimeMode=" + tensorRuntimeMode + '\n'
                + "gpuPrefill=" + gpuPrefillEnabled + '\n'
                + "gpuDecode=" + gpuDecodeEnabled + '\n'
                + "gpuDecodeAttention=" + gpuDecodeAttentionEnabled + '\n'
                + "gpuDiffusionBlockProjection=" + gpuDiffusionBlockProjectionEnabled + '\n'
                + "packedBlockAttention=" + packedBlockAttentionEnabled + '\n'
                + "packedPrefill=" + packedPrefillEnabled + '\n'
                + "generationOptions=" + generationOptions + '\n'
                + "groupedDecodeQkvSplitSize=" + groupedDecodeQkvSplitSize + '\n'
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
