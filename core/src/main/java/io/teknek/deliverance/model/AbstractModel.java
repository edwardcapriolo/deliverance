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
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.*;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
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
    private static final Logger logger = LoggerFactory.getLogger(AbstractModel.class);

    private static final Integer MAX_BATCH_SIZE = Integer.getInteger("jlama.max_batch_size", 256);

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
    protected EmbedInput embedInput;
    protected SampleOutput sampleOutput;
    protected TransformerBlock[] transformerBlocks;
    protected KvBufferCache kvBufferCache;
    protected final ConfigurableTensorProvider configurableTensorProvider;
    protected final MetricRegistry metricRegistry;
    protected final TensorAllocator tensorAllocator;

    //embedding
    protected Optional<PoolingLayer> poolingLayer;

    protected final ToolCallParser toolCallParser;

    protected ClassifyOutput classifyOutput;
    protected WrappedForkJoinPool pool;
    private volatile Consumer<GenerationDebugEvent> generationDebugHook = event -> {};

    protected AbstractModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer t, DType workingMemoryDType,
                            DType workingMemoryQType, Optional<DType> modelQType, ConfigurableTensorProvider provider,
                            MetricRegistry metricRegistry, TensorAllocator tensorAllocator, KvBufferCacheSettings kvBufferCacheSettings,
                            ToolCallParser toolCallParser, WrappedForkJoinPool pool) {
        this.inferenceType = inferenceType;
        this.config = c;
        this.weights = w;
        this.tokenizer = t;

        this.modelDType = w.getModelDType();
        this.workingDType = workingMemoryDType;
        this.modelQType = modelQType;
        this.kvBufferCache = new KvBufferCache(this, kvBufferCacheSettings);
        this.configurableTensorProvider = provider;
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
        this.embedInput = inferenceType.isInput ? loadInputWeights() : null;
        this.transformerBlocks = inferenceType.isFwdPass ? loadTransformerBlockWeights() : null;
        this.sampleOutput = inferenceType.isOutput ? loadOutputWeights() : null;
        this.classifyOutput = inferenceType.isClassify ? loadClassifierWeights() : null;
        this.poolingLayer = inferenceType.isPooling ? Optional.ofNullable(loadPoolingWeights()) : Optional.empty();
        this.pool = pool;
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

    protected abstract EmbedInput loadInputWeights();
    protected abstract SampleOutput loadOutputWeights();
    protected abstract TransformerBlock[] loadTransformerBlockWeights();

    @Override
    public void close() {
        kvBufferCache.close();
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

    /**
     *
     * @return Some if the tokenizer inside this model has a chat_template/prompt template Empty if not.
     */
    public Optional<PromptSupport> promptSupport() {
        return tokenizer.chatTemplate().map(template -> new PromptSupport(
                Map.of("default", template),
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

    SamplerReturn createNextToken(GeneratorParameters generatorParameters, AbstractTensor logits, AbstractTensor last,
                        ResponseContext responseContext, Random random, float temperature){
        if (generatorParameters.guidedChoice.isPresent()) {
            GuidedChoiceSampler sampler = new GuidedChoiceSampler(this, last.slice(last.shape().first() - 1),
                    logits, sampleOutput.getOutputLayerNorm(), generatorParameters.guidedChoice.get(), responseContext);
           return new SamplerReturn(sampler.sample());
        } else {
            DeliveranceSampler legacy = new DeliveranceSampler(this, generatorParameters,
                    last.slice(last.shape().first() -1), logits, sampleOutput.getOutputLayerNorm(), random, random.nextFloat());
            return legacy.sample();
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

    private static double nanosToMs(long nanos) {
        return nanos / 1_000_000.0;
    }

    private Response withGenerationTiming(Response response, long generationStartNanos, long timeToFirstTokenNanos) {
        double totalTimeMs = nanosToMs(System.nanoTime() - generationStartNanos);
        double timeToFirstTokenMs = nanosToMs(timeToFirstTokenNanos);
        int generatedTokenCount = response.generatedTokens == null ? 0 : response.generatedTokens.size();
        double avgTimePerTokenMs = generatedTokenCount == 0 ? 0.0 : totalTimeMs / generatedTokenCount;
        return response.copyWithTiming(timeToFirstTokenMs, avgTimePerTokenMs, totalTimeMs);
    }

    private Response buildTimedResponse(
            FinishReason reason,
            int promptLength,
            ResponseContext responseContext,
            long generationStartNanos,
            long timeToFirstTokenNanos
    ) {
        return postProcessResponse(withGenerationTiming(new Response(
                        responseContext.responseText.toString(),
                        responseContext.responseTextWithSpecialTokens.toString(),
                        reason,
                        promptLength,
                        responseContext.generatedTokens,
                        0,
                        0,
                        responseContext.samplerReturnList),
                generationStartNanos,
                timeToFirstTokenNanos));
    }

    private Optional<Response> maybeStopAfterToken(GeneratorParameters generatorParameters, ResponseContext responseContext,
                                                   long[] encoded, int promptLength, int next,
                                                   long generationStartNanos, long timeToFirstTokenNanos) {
        if (generatorParameters.maxTokens.isPresent()) {
            if (responseContext.generatedTokens.size() >= generatorParameters.maxTokens.get()) {
                FinishReason reason = FinishReason.MAX_TOKENS;
                return Optional.of(buildTimedResponse(reason, promptLength, responseContext, generationStartNanos, timeToFirstTokenNanos));
            }
        }
        if (generatorParameters.guidedChoice.isPresent()) {
            if (generatorParameters.guidedChoice.get().contains(responseContext.responseText.toString())) {
                FinishReason reason = FinishReason.STOP_TOKEN;
                return Optional.of(buildTimedResponse(reason, promptLength, responseContext, generationStartNanos, timeToFirstTokenNanos));
            }
        }
        Optional<Response> shouldEnd = stopWords(generatorParameters, responseContext, encoded.length);
        if (shouldEnd.isPresent()) {
            return Optional.of(postProcessResponse(withGenerationTiming(shouldEnd.get(), generationStartNanos, timeToFirstTokenNanos)));
        }
        Optional<Response> shouldEndTools = getToolCallParser().shouldEndTurn(responseContext, encoded.length);
        if (shouldEndTools.isPresent()) {
            return Optional.of(postProcessResponse(withGenerationTiming(shouldEndTools.get(), generationStartNanos, timeToFirstTokenNanos)));
        }
        if (config.eosTokens.contains(next)) {
            FinishReason reason = FinishReason.STOP_TOKEN;
            return Optional.of(buildTimedResponse(reason, promptLength, responseContext, generationStartNanos, timeToFirstTokenNanos));
        }
        return Optional.empty();
    }

    @Override
    public Response generate(UUID sessionId, PromptContext promptContext, GeneratorParameters generatorParameters,
                                     GenerateEvent eventFired) {
        long generationStartNanos = System.nanoTime();
        long timeToFirstTokenNanos = 0L;

        ResponseContext responseContext = new ResponseContext(this);
        Random random = generatorParameters.seed.map(Random::new).orElseGet(Random::new);
        long[] encoded = encodeText(promptContext.getPrompt());

        //long [] encoded = this.preTrainedTokenizer.encode(promptContext.getPrompt(), EncodeOptions.defaults());

        //can we get pos token from tokziers
        if (encoded.length > 0 && encoded[0] == config.bosToken) {
            logger.warn("encoded [] started with BOS token removing it");
            encoded = Arrays.copyOfRange(encoded, 1, encoded.length);
        }
        int ntokens = generatorParameters.ntokens.orElse(config.contextLength);
        Preconditions.checkArgument(encoded.length < config.contextLength
                && encoded.length < ntokens, "Prompt exceeds ntokens");
        if (ntokens > config.contextLength) {
            throw new GenerationException(String.format("ntokens %d exceed config length %d",  ntokens, config.contextLength));
        }
        float temperature = generatorParameters.temperature.orElse(0.0f);
        try (KvBufferCache.KvBuffer kvmem = kvBufferCache.getEphemeralKvBuffer()) {
            try (AbstractTensor logits = makeDenseTensor(config.vocabularySize)) {
                int [] promptTokens = constructPromptTokens(encoded);

                // Prefix cache only changes how prompt KV state is obtained. It must not change prompt length,
                // decode start, or generation budget. Exact generated text equivalence is a stronger
                // batch/chunk-invariance property and is not assumed here.
                KvBufferCache.PrefixEntry prefixHit = kvBufferCache.lookupPrefix(promptTokens, generatorParameters.cacheSalt);
                int prefixLength = 0;
                if (prefixHit != null) {
                    prefixLength = prefixHit.length();
                    kvBufferCache.copyPrefix(prefixHit.buffer(), kvmem, prefixLength);
                    generationDebugHook.accept(new GenerationDebugEvent(
                            GenerationDebugEventType.AFTER_PREFIX_COPY,
                            promptTokens,
                            prefixLength,
                            prefixLength,
                            promptTokens.length - prefixLength,
                            kvmem));
                }
                GenerationCursor cursor = GenerationCursor.from(promptTokens, prefixLength);
                int startPos = cursor.startPosition();
                kvmem.setCurrentContextPosition(startPos);
                int [] tokensToProcess = cursor.tokensToProcess();
                AbstractTensor last;
                if (cursor.hasTokensToProcess()) {
                    last = batchForward(tokensToProcess, startPos, kvmem);
                    kvBufferCache.storePrefix(promptTokens, kvmem, generatorParameters.cacheSalt);
                } else {
                    last = forward(cursor.replayToken(), cursor.replayPosition(), kvmem);
                }

                generationDebugHook.accept(new GenerationDebugEvent(
                        GenerationDebugEventType.AFTER_PROMPT_PREFILL,
                        promptTokens,
                        prefixLength,
                        startPos,
                        tokensToProcess.length,
                        kvmem));


                SamplerReturn nextSamplerRet = createNextToken(generatorParameters, logits, last, responseContext, random, temperature);
                int next = nextSamplerRet.token;
                last.close();
                responseContext.add(nextSamplerRet, eventFired);
                timeToFirstTokenNanos = System.nanoTime() - generationStartNanos;
                this.metricRegistry.timer("generation.time_to_first_token").update(timeToFirstTokenNanos, TimeUnit.NANOSECONDS);
                logger.info("time_to_first_token={} prefix_length={}", timeToFirstTokenNanos / 1_000_000.0 , prefixLength);
                Optional<Response> firstStop = maybeStopAfterToken(generatorParameters, responseContext, encoded, encoded.length, next,
                        generationStartNanos, timeToFirstTokenNanos);
                if (firstStop.isPresent()) {
                    return withGenerationTiming(firstStop.get(), generationStartNanos, timeToFirstTokenNanos);
                }
                for (int i = cursor.decodeStartPosition(); i < ntokens; i++) {
                    AbstractTensor output = forward(next, i, kvmem);
                    //reuse next to save memory
                    SamplerReturn nextSample = createNextTokenLoop(generatorParameters, output, logits, responseContext, random, temperature);
                    next = nextSample.token;
                    output.close();
                    kvmem.incrementContextPosition();
                    responseContext.add(nextSample, eventFired);

                    Optional<Response> stop = maybeStopAfterToken(generatorParameters, responseContext, encoded, encoded.length, next,
                            generationStartNanos, timeToFirstTokenNanos);
                    if (stop.isPresent()) {
                        return withGenerationTiming(stop.get(), generationStartNanos, timeToFirstTokenNanos);
                    }

                }
            }
        }

        return withGenerationTiming(postProcessResponse(new Response(
                        responseContext.responseText.toString(),
                        responseContext.responseTextWithSpecialTokens.toString(),
                        FinishReason.MAX_TOKENS,
                        encoded.length,
                        responseContext.generatedTokens,
                        0,
                        0,
                        responseContext.samplerReturnList)),
                generationStartNanos,
                timeToFirstTokenNanos);
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

    public AbstractTensor batchForward(int[] token_ids, int startPos, KvBufferCache.KvBuffer kvbuf,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        AbstractTensor embedding = null;

        CausualWhisperer.LOGGER.debug("batchForward from 0 to token_ids.length {} max_batch_size {} per iteration",
                token_ids.length, MAX_BATCH_SIZE);
        for (int i = 0; i < token_ids.length; i += MAX_BATCH_SIZE) {
            int[] batch = Arrays.copyOfRange(token_ids, i, Math.min(token_ids.length, i + MAX_BATCH_SIZE));
            embedding = embedInput.batchInputsToEmbeddings(batch, startPos + i);
            embedding = forward(embedding, startPos + i, kvbuf, tensorReducer);
        }
        return embedding;
    }

    protected AbstractTensor forward(int token_id, int pos, KvBufferCache.KvBuffer kvbuf) {
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
        AbstractTensor embedding = embedInput.inputTokenToEmbedding(token_id, pos);
        return forward(embedding, pos, kvbuf, tensorReducer);
    }


    public AbstractTensor forward(AbstractTensor embedding, int startPos, KvBufferCache.KvBuffer kvbuf,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        for (int i = config.dctx().layerStart; i < config.dctx().layerEnd; i++) {
            int relativeLayer = i - config.dctx().layerStart;
            AbstractTensor ref = embedding; // reference so we can free
            embedding = transformerBlocks[relativeLayer].forward(embedding, startPos, kvbuf, tensorReducer);
            ref.close();
        }
        return embedding;
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
