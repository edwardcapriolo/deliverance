package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;

import java.nio.FloatBuffer;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;


import io.teknek.deliverance.CausualWhisperer;
import io.teknek.deliverance.Classifier;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.classifier.ClassifyOutput;
import io.teknek.deliverance.embedding.PoolingLayer;

import io.teknek.deliverance.embedding.PoolingType;
import io.teknek.deliverance.generator.*;
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
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.*;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.TensorOperations;
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
    protected final ConfigurableTensorProvider configurableTensorProvider;
    protected final MetricRegistry metricRegistry;
    protected final TensorAllocator tensorAllocator;
    protected final TensorParallelContext tensorParallelContext;
    protected final TensorParallelCollectives tensorParallelCollectives;
    private final EnumMap<TensorProviderKind, TensorOperations> tensorOperations = new EnumMap<>(TensorProviderKind.class);
    private boolean tensorProviderExplicit;
    private GossipParallelMembership gossipParallelMembership;

    //embedding
    protected Optional<PoolingLayer> poolingLayer;

    protected final ToolCallParser toolCallParser;

    protected ClassifyOutput classifyOutput;
    protected WrappedForkJoinPool pool;
    protected PreTrainedTokenizer preTrainedTokenizer;
    protected int maxBatchSize = DEFAULT_MAX_BATCH_SIZE;
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
        this.kvBufferCache = new KvBufferCache(this, kvBufferCacheSettings);
        this.configurableTensorProvider = provider;
        this.tensorOperations.put(TensorProviderKind.SIMD, provider.get());
        this.metricRegistry = metricRegistry;
        this.tensorAllocator = tensorAllocator;
        this.toolCallParser = toolCallParser;

        if (workingMemoryQType == null) {
            workingMemoryQType = configurableTensorProvider.get().preferredWorkingQuantizedType();
        }

        // FIXME: This is a hack to support Avoid Q8F32 evals
        if (modelDType == DType.F32 && workingMemoryQType != DType.F32 && modelQType.isEmpty()) {
            workingMemoryQType = DType.F32;
        }

        // FIXME: This is a hack to support Avoid Q8BF16 evals
        if (modelDType == DType.BF16 && workingMemoryQType != DType.BF16 && workingMemoryQType != DType.F32 && modelQType.isEmpty()) {
            workingMemoryQType = DType.BF16;
        }

        // Check to make sure the model is big enough to support Q4I8 computations
        // If not, fall back to F32
        if (modelDType == DType.Q4
                && workingMemoryQType == DType.I8
                && ((c.embeddingLength / Q8ByteBufferTensor.BLOCK_SIZE) % (FloatVector.SPECIES_PREFERRED.vectorBitSize() / Float.SIZE) != 0
                || (c.hiddenLength / Q8ByteBufferTensor.BLOCK_SIZE) % (FloatVector.SPECIES_PREFERRED.vectorBitSize() / Float.SIZE) != 0)) {
            workingMemoryQType = DType.F32;
        }

        // Check to make sure the model is big enough to support Q4I8 computations
        // If not, fall back to F32
        if (modelDType == DType.Q4
                && workingMemoryQType == DType.I8
                && (c.embeddingLength / Q8ByteBufferTensor.BLOCK_SIZE) % (FloatVector.SPECIES_PREFERRED.vectorBitSize() / Float.SIZE) != 0) {
            logger.warn("Determined model could not support quant type. Request {} model {} falling back to {} ",
                    workingMemoryQType, modelDType, DType.F32 );
            workingMemoryQType = DType.F32;
        }

        // Some operation providers don't support Q4I8
        if (modelDType == DType.Q4 && workingMemoryQType.size() < configurableTensorProvider.get().preferredWorkingQuantizedType().size()) {
            workingMemoryQType = configurableTensorProvider.get().preferredWorkingQuantizedType();
            logger.warn("Tensor provider {} does not support Q4. Using {} as workingMemoryType ",
                    configurableTensorProvider.get().name(), workingMemoryQType);
        }

        if (workingMemoryQType != workingMemoryDType) {
            boolean supportsQType;
            AbstractTensor tmp = makeDenseTensor(Q8ByteBufferTensor.BLOCK_SIZE);
            try (AbstractTensor tmp2 = configurableTensorProvider.get().quantize(tmp, workingMemoryQType, 0, Q8ByteBufferTensor.BLOCK_SIZE)) {
                supportsQType = tmp2.dType() == workingMemoryQType;
                if (!supportsQType) {
                    logger.warn("Quantized memory type {} not supported, falling back to {}", workingMemoryQType, workingMemoryDType);
                    this.workingQType = this.workingDType;
                } else {
                    this.workingQType = workingMemoryQType;
                }
            }
        } else {
            this.workingQType = workingMemoryQType;
        }

        logger.info("Tensor provider = {}, parallelSplitSize = {} ",
                configurableTensorProvider.get().name(), configurableTensorProvider.get().parallelSplitSize());
        logger.info("Model type = {}, Working memory type = {}, Quantized memory type = {}", modelDType, workingDType,
                workingQType);
        logger.debug("model constructor phase=input_weights enabled={}", inferenceType.isInput);
        this.embedInput = inferenceType.isInput ? loadInputWeights() : null;
        logger.debug("model constructor phase=input_weights done enabled={}", inferenceType.isInput);
        logger.debug("model constructor phase=transformer_blocks enabled={}", inferenceType.isFwdPass);
        this.transformerBlocks = inferenceType.isFwdPass ? loadTransformerBlockWeights() : null;
        logger.debug("model constructor phase=transformer_blocks done enabled={} layers={}", inferenceType.isFwdPass,
                this.transformerBlocks == null ? 0 : this.transformerBlocks.length);
        logger.debug("model constructor phase=output_weights enabled={}", inferenceType.isOutput);
        this.sampleOutput = inferenceType.isOutput ? loadOutputWeights() : null;
        logger.debug("model constructor phase=output_weights done enabled={}", inferenceType.isOutput);
        logger.debug("model constructor phase=classifier_weights enabled={}", inferenceType.isClassify);
        this.classifyOutput = inferenceType.isClassify ? loadClassifierWeights() : null;
        logger.debug("model constructor phase=classifier_weights done enabled={}", inferenceType.isClassify);
        this.poolingLayer = inferenceType.isPooling ? Optional.ofNullable(loadPoolingWeights()) : Optional.empty();
        this.pool = pool;
        logger.debug("model constructor complete config={} inference_type={}", config.getClass().getSimpleName(), inferenceType);
    }

    void addTensorOperations(Map<TensorProviderKind, TensorOperations> additionalTensorOperations) {
        this.tensorOperations.putAll(additionalTensorOperations);
    }

    void setTensorProviderExplicit(boolean tensorProviderExplicit) {
        this.tensorProviderExplicit = tensorProviderExplicit;
    }

    public boolean isTensorProviderExplicit() {
        return tensorProviderExplicit;
    }

    public Optional<TensorOperations> tensorOperations(TensorProviderKind kind) {
        return Optional.ofNullable(tensorOperations.get(kind));
    }

    public TensorOperations primaryTensorOperations() {
        return configurableTensorProvider.get();
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

    protected String decodeToken(int token) {
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
                                  ResponseContext responseContext, Random random, float temperature){
        try (AbstractTensor lastTokenOutput = last.copyLastTokenOutput(tensorAllocator)) {
            if (generatorParameters.guidedChoice.isPresent()) {
                GuidedChoiceSampler sampler = new GuidedChoiceSampler(this, lastTokenOutput,
                        logits.tensor(), sampleOutput.getOutputLayerNorm(), generatorParameters.guidedChoice.get(), responseContext);
               return new SamplerReturn(sampler.sample());
            } else {
                DeliveranceSampler legacy = new DeliveranceSampler(this, generatorParameters,
                        lastTokenOutput, logits.tensor(), sampleOutput.getOutputLayerNorm(), random, random.nextFloat());
                return legacy.sample();
            }
        }
    }

    SamplerReturn createNextTokenLoop(GeneratorParameters generatorParameters, AbstractTensor output,
                            AbstractTensor logits, ResponseContext responseContext, Random random, float temperature){
        if (generatorParameters.guidedChoice.isPresent()) {
            GuidedChoiceSampler sampler1 = new GuidedChoiceSampler(this, output, logits,
                    sampleOutput.getOutputLayerNorm(), generatorParameters.guidedChoice.get(), responseContext);
            //TODO should guided choice have logits how expesnive is two code paths going forward
            return new SamplerReturn(sampler1.sample());
        } else {
            DeliveranceSampler legacy = new DeliveranceSampler(this, generatorParameters, output, logits,
                    sampleOutput.getOutputLayerNorm(), random, random.nextFloat());
            return legacy.sample();
        }
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
        VectorTensorMathUtils.softMax(scores, 0, classes));
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
                    int[] batch = Arrays.copyOfRange(token_ids, i, Math.min(token_ids.length, i + maxBatchSize));
                    progress.chunkStart = i;
                    progress.chunkTokens = batch.length;
                    AbstractTensor inputEmbeddings = embedInput.batchInputsToEmbeddings(batch, startPos + i);
                    lastBatchOutput = forward(inputEmbeddings, startPos + i, kvbuf, tensorReducer);
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
        return forward(embedding, pos, kvbuf, tensorReducer);
        }
    }


    public AbstractTensor forward(AbstractTensor embedding, int startPos, KvBufferCache.KvBuffer kvbuf,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        try (Timer.Context ignored = InferenceProfiler.timer(metricRegistry, "abstractmodel.forward_layers").time()) {
        emitLayerDebug(-1, "input", embedding);
        int batchTokens = embedding.shape().first();
        for (int i = 0; i < config.numberOfLayers; i++) {
            int relativeLayer = i;
            AbstractTensor ref = embedding; // reference so we can free
            embedding = transformerBlocks[relativeLayer].forward(embedding, startPos, kvbuf, tensorReducer);
            emitLayerDebug(relativeLayer, "layer_output", embedding);
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

    /** This is a hook method that does nothing here but can be overridden by subclasses */
    public AbstractTensor maybeQuantize(AbstractTensor t) {
        AbstractTensor t2 = tensorAllocator.getDirty(t.dType(), t.shape());
        t2.copyFrom(t, 0, 0, Ints.checkedCast(t.size()));
        return t2;
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

}
