package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import com.codahale.metrics.Timer;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;

import java.nio.FloatBuffer;
import java.util.*;
import java.util.function.Consumer;


import io.teknek.deliverance.CausualWhisperer;
import io.teknek.deliverance.Classifier;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.classifier.ClassifyOutput;
import io.teknek.deliverance.embedding.PoolingLayer;
import io.teknek.deliverance.embedding.PoolingType;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.math.VectorMathUtils;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.safetensors.prompt.ToolCall;
import io.teknek.deliverance.tensor.*;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tokenizer.Tokenizer;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import jdk.incubator.vector.FloatVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static io.teknek.deliverance.tensor.DebugSupport.debug;


public abstract class AbstractModel implements Generator, Classifier {
    private static final Logger logger = LoggerFactory.getLogger(AbstractModel.class);

    private static final Integer MAX_BATCH_SIZE = Integer.getInteger("jlama.max_batch_size", 256);

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
    protected final Tokenizer tokenizer;
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
    protected final TensorCache tensorCache;

    //embedding
    protected Optional<PoolingLayer> poolingLayer;

    protected final TokenRenderer tokenRenderer;
    protected final ToolCallParser toolCallParser;

    protected ClassifyOutput classifyOutput;
    protected WrappedForkJoinPool pool;

    protected AbstractModel(InferenceType inferenceType, Config c, WeightLoader w, Tokenizer t, DType workingMemoryDType,
                            DType workingMemoryQType, Optional<DType> modelQType, ConfigurableTensorProvider provider,
                            MetricRegistry metricRegistry, TensorCache tensorCache, KvBufferCacheSettings kvBufferCacheSettings,
                            TokenRenderer tokenRenderer, ToolCallParser toolCallParser, WrappedForkJoinPool pool) {
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
        this.tensorCache = tensorCache;
        this.tokenRenderer = tokenRenderer;
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
            workingMemoryQType = DType.F32;
        }

        // Some operation providers don't support Q4I8
        if (modelDType == DType.Q4 && workingMemoryQType.size() < configurableTensorProvider.get().preferredWorkingQuantizedType().size()) {
            workingMemoryQType = configurableTensorProvider.get().preferredWorkingQuantizedType();
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
        return tensorCache.get(workingDType, s);
    }

    public AbstractTensor makeDenseTensor(int... shape) {
        return tensorCache.get(workingDType, TensorShape.of(shape));
    }

    public AbstractTensor makeDenseTensor(TensorShape s) {
        return tensorCache.get(workingDType, s);
    }

    public DType getWorkingDType() {
        return workingDType;
    }

    /**
     *
     * @return Some if the tokenizer inside this model has a chat_template/prompt template Empty if not.
     */
    public Optional<PromptSupport> promptSupport() {
        return tokenizer.promptSupport();
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

    public class ResponseContext {
        public final StringBuilder responseText = new StringBuilder();
        public final StringBuilder responseTextWithSpecialTokens = new StringBuilder();
        public final List<Integer> generatedTokens = new ArrayList<>();

        public void add(int token, GenerateEvent event){
            generatedTokens.add(token);
            String decoded = tokenizer.decode(token);
            String cleaned = tokenRenderer.tokenizerToRendered(decoded);
            if (tokenizer.getModel().isSpecialToken(token)) {
                responseTextWithSpecialTokens.append(cleaned);
            } else {
                event.emit(token, decoded, cleaned, 0);
                responseText.append(cleaned);
                responseTextWithSpecialTokens.append(cleaned);
            }
        }

        public StringBuilder getResponseTextWithSpecialTokens() {
            return this.responseTextWithSpecialTokens;
        }

        public StringBuilder getResponseText() {
            return responseText;
        }

        public List<Integer> getGeneratedTokens() {
            return generatedTokens;
        }
    }

    int createNextToken(GeneratorParameters generatorParameters, AbstractTensor logits, AbstractTensor last,
                        ResponseContext responseContext, Random random, float temperature){
        if (generatorParameters.guidedChoice.isPresent()) {
            GuidedChoiceSampler sampler = new GuidedChoiceSampler(this, last.slice(last.shape().first() - 1),
                    logits, sampleOutput.getOutputLayerNorm(), tokenizer, generatorParameters.guidedChoice.get(), responseContext.responseText);
           return sampler.sample();
        } else {
            GeneratorSampler sampler = new GeneratorSampler(this, last.slice(last.shape().first() - 1), temperature,
                    random.nextFloat(), logits, sampleOutput.getOutputLayerNorm(), false, 0);
            return sampler.sample();
        }
    }

    int createNextTokenLoop(GeneratorParameters generatorParameters, AbstractTensor output,
                            AbstractTensor logits, ResponseContext responseContext, Random random, float temperature){
        if (generatorParameters.guidedChoice.isPresent()) {
            GuidedChoiceSampler sampler1 = new GuidedChoiceSampler(this, output, logits,
                    sampleOutput.getOutputLayerNorm(), tokenizer, generatorParameters.guidedChoice.get(), responseContext.responseText);
            return sampler1.sample();
        } else {
            GeneratorSampler sampler1 = new GeneratorSampler(this, output, temperature,
                    random.nextFloat(), logits, sampleOutput.getOutputLayerNorm(), false, 0);
            return sampler1.sample();
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
                                reason, promptLength, responseContext.generatedTokens, 0, 0));
                    } else {
                        int index = responseContext.responseTextWithSpecialTokens.indexOf(stop);
                        responseContext.responseTextWithSpecialTokens.delete(index, responseContext.responseTextWithSpecialTokens.length());
                        int x = responseContext.responseText.indexOf(stop);
                        if (x != -1) {
                            responseContext.responseText.delete(x, responseContext.responseText.length());
                        }
                        return Optional.of(new Response(responseContext.responseText.toString(), responseContext.responseTextWithSpecialTokens.toString(),
                                reason, promptLength, responseContext.generatedTokens, 0, 0));
                    }
                }
            }
        }
        return Optional.empty();
    }

    @Override
    public Response generate(UUID sessionId, PromptContext promptContext, GeneratorParameters generatorParameters,
                                     GenerateEvent eventFired) {
        ResponseContext responseContext = new ResponseContext();
        Random random = generatorParameters.seed.map(Random::new).orElseGet(Random::new);
        long[] encoded = tokenizer.encode(promptContext.getPrompt());
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
        try (KvBufferCache.KvBuffer kvmem = kvBufferCache.getKvBuffer(sessionId.toString())) {
            int startPos = kvmem.getCurrentContextPosition();
            try (AbstractTensor logits = makeDenseTensor(config.vocabularySize)) {
                int [] promptTokens = constructPromptTokens(encoded);
                AbstractTensor last = batchForward(promptTokens, startPos, kvmem);
                int next = createNextToken(generatorParameters, logits, last, responseContext, random, temperature);
                last.close();
                responseContext.add(next, eventFired);
                //here we have added the first token, consider checking stop conditions here
                for (int i = startPos + promptTokens.length; i < ntokens; i++) {
                    AbstractTensor output = forward(next, i, kvmem);
                    //reuse next to save memory
                    next = createNextTokenLoop(generatorParameters, output, logits, responseContext, random, temperature);
                    output.close();
                    kvmem.incrementContextPosition();
                    responseContext.add(next, eventFired);

                    if (generatorParameters.maxTokens.isPresent()){
                        if (responseContext.generatedTokens.size() >= generatorParameters.maxTokens.get()) {
                            FinishReason reason = FinishReason.MAX_TOKENS;
                            return new Response(responseContext.responseText.toString(), responseContext.responseTextWithSpecialTokens.toString(),
                                    reason,  encoded.length, responseContext.generatedTokens, 0, 0);
                        }
                    }
                    if (generatorParameters.guidedChoice.isPresent()) {
                        if (generatorParameters.guidedChoice.get().contains(responseContext.responseText.toString())) {
                            FinishReason reason = FinishReason.STOP_TOKEN;
                            return new Response(responseContext.responseText.toString(), responseContext.responseTextWithSpecialTokens.toString(),
                                    reason,  encoded.length, responseContext.generatedTokens, 0, 0);
                        }
                    }
                    Optional<Response> shouldEnd = stopWords(generatorParameters, responseContext, encoded.length);
                    if (shouldEnd.isPresent()) {
                        return shouldEnd.get();
                    }
                    if (config.eosTokens.contains(next)){
                        FinishReason reason = FinishReason.STOP_TOKEN;
                        return new Response(responseContext.responseText.toString(), responseContext.responseTextWithSpecialTokens.toString(),
                                reason,  encoded.length, responseContext.generatedTokens, 0, 0);
                    }
                    Optional<Response> shouldEndTools = getToolCallParser().shouldEndTurn(responseContext, encoded.length);
                    if (shouldEndTools.isPresent()) {
                        return shouldEndTools.get();
                    }

                }
            }
        }

        return  new Response(responseContext.responseText.toString(), responseContext.responseTextWithSpecialTokens.toString(),
                FinishReason.MAX_TOKENS, encoded.length, responseContext.generatedTokens, 0, 0);
    }

    /*
    public Response generate3(UUID sessionId, PromptContext promptContext, GeneratorParameters generatorParameters,
                             GenerateEvent onTokenWithTimings) {
        Random random = generatorParameters.seed.map(Random::new).orElseGet(Random::new);
        long[] encoded = tokenizer.encode(promptContext.getPrompt());
        if (encoded.length > 0 && encoded[0] == config.bosToken) {
            encoded = Arrays.copyOfRange(encoded, 1, encoded.length);
        }
        int ntokens = generatorParameters.ntokens.orElse(config.contextLength);
        Preconditions.checkArgument(encoded.length < config.contextLength
                && encoded.length < ntokens, "Prompt exceeds max tokens");
        if (ntokens > config.contextLength) {
            throw new GenerationException(String.format("ntokens %d exceed config length %d",  ntokens, config.contextLength));
        }
        float temperature = generatorParameters.temperature.orElse(0.0f);

        try (KvBufferCache.KvBuffer kvmem = kvBufferCache.getKvBuffer(sessionId.toString())) { // k and v for context window
            int startPos = kvmem.getCurrentContextPosition(); // Number of tokens in the buffer
            FinishReason reason = FinishReason.MAX_TOKENS;
            int promptLength;
            long promptBatchTime;
            StringBuilder responseText = new StringBuilder();
            StringBuilder responseTextWithSpecialTokens = new StringBuilder();
            ArrayList<Integer> generatedTokens = new ArrayList<>();

            try (AbstractTensor logits = makeDenseTensor(config.vocabularySize)) {
                int [] promptTokens = constructPromptTokens(encoded);
                promptLength = encoded.length;
                long start = System.currentTimeMillis();
                AbstractTensor last = batchForward(promptTokens, startPos, kvmem);
                logger.debug("After batch forward size: {} shape: {}" , last.size(), last.shape());
                promptBatchTime = System.currentTimeMillis() - start;
                float batchMsPerToken = Math.round((((double) promptBatchTime) / (double) promptLength));
                int next = Integer.MIN_VALUE;
                if (generatorParameters.guidedChoice.isPresent()) {
                    GuidedChoiceSampler sampler = new GuidedChoiceSampler(this, last.slice(last.shape().first() - 1),
                            logits, sampleOutput.getOutputLayerNorm(), tokenizer, generatorParameters.guidedChoice.get(), responseText);
                    next = sampler.sample();
                } else {
                    GeneratorSampler sampler = new GeneratorSampler(this, last.slice(last.shape().first() - 1), temperature,
                            random.nextFloat(), logits, sampleOutput.getOutputLayerNorm());
                    next = sampler.sample();
                }
                generatedTokens.add(next);
                float genMsPerToken = 0;

                last.close();
                String decoded = tokenizer.decode(next);
                String cleaned = tokenRenderer.tokenizerToRendered(decoded);
                if (tokenizer.getModel().isSpecialToken(next)) {
                    responseTextWithSpecialTokens.append(cleaned);
                } else {
                    onTokenWithTimings.emit(next, decoded, cleaned, batchMsPerToken);
                    responseText.append(cleaned);
                    responseTextWithSpecialTokens.append(cleaned);
                }
                //AT this point the response text could be a stop word or a guided choice consider stopping here
                start = System.currentTimeMillis();
                for (int i = startPos + promptTokens.length; i < ntokens; i++) {
                    AbstractTensor output = forward(next, i, kvmem);

                    if (generatorParameters.guidedChoice.isPresent()) {
                        GuidedChoiceSampler sampler1 = new GuidedChoiceSampler(this, output, logits, sampleOutput.getOutputLayerNorm(), tokenizer, generatorParameters.guidedChoice.get(), responseText);
                        next = sampler1.sample();
                    } else {
                        GeneratorSampler sampler1 = new GeneratorSampler(this, output, temperature, random.nextFloat(), logits, sampleOutput.getOutputLayerNorm());
                        next = sampler1.sample();
                    }
                    generatedTokens.add(next);
                    output.close();
                    kvmem.incrementContextPosition();
                    if (config.eosTokens.contains(next)) {
                        reason = FinishReason.STOP_TOKEN;
                        break;
                    }
                    String decoded1 = tokenizer.decode(next);
                    String cleaned1 = tokenRenderer.tokenizerToRendered(decoded1);
                    if (tokenizer.getModel().isSpecialToken(next)) {
                        responseTextWithSpecialTokens.append(cleaned1);
                    } else {
                        genMsPerToken = (System.currentTimeMillis() - start) / (float) generatedTokens.size();
                        onTokenWithTimings.emit(next, decoded1, cleaned1, genMsPerToken);
                        responseTextWithSpecialTokens.append(cleaned1);
                        responseText.append(cleaned1);
                    }
                    if (generatorParameters.maxTokens.isPresent()){
                        if (generatedTokens.size() >= generatorParameters.maxTokens.get()) {
                            reason = FinishReason.MAX_TOKENS;
                            return new Response(responseText.toString(), responseTextWithSpecialTokens.toString(),
                                    reason, promptLength, generatedTokens, promptBatchTime, System.currentTimeMillis() - start);
                        }
                    }
                    if (generatorParameters.guidedChoice.isPresent()) {
                        if (generatorParameters.guidedChoice.get().contains(responseText.toString())) {
                            reason = FinishReason.STOP_TOKEN;
                            break;
                        }
                    }
                    if (generatorParameters.stopWords.isPresent()){
                        List<String> stops = generatorParameters.stopWords.get();
                        for (String stop: stops){
                            if (responseTextWithSpecialTokens.indexOf(stop) != -1) {
                                reason = FinishReason.STOP_TOKEN;
                                if (generatorParameters.includeStopStrInOutput.isPresent() && generatorParameters.includeStopStrInOutput.get()){
                                    long end = System.currentTimeMillis();
                                    return new Response(responseText.toString(), responseTextWithSpecialTokens.toString(),
                                        reason, promptLength, generatedTokens, promptBatchTime, end - start);
                                } else {
                                    long end = System.currentTimeMillis();
                                    int index = responseTextWithSpecialTokens.indexOf(stop);
                                    responseTextWithSpecialTokens.delete(index, responseTextWithSpecialTokens.length());
                                    int x = responseText.indexOf(stop);
                                    responseText.delete(x, responseText.length());
                                    return new Response(responseText.toString(), responseTextWithSpecialTokens.toString(),
                                            reason, promptLength, generatedTokens, promptBatchTime, end - start);
                                }
                            }
                        }
                    }
                }

                long end = System.currentTimeMillis();
                //post process response is still missing
                return new Response(responseText.toString(), responseTextWithSpecialTokens.toString(),
                        reason, promptLength, generatedTokens, promptBatchTime, end - start);
            }
        }
    } */



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
        AbstractTensor scores = tensorCache.getDirty(workingDType, TensorShape.of(classes));
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
        int[] encoded = Arrays.stream(tokenizer.encode(input)).mapToInt(Ints::checkedCast).toArray();
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
                    AbstractTensor pooled = tensorCache.getDirty(workingDType, TensorShape.of(1, config.embeddingLength));
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
        AbstractTensor t2 = tensorCache.getDirty(t.dType(), t.shape());
        t2.copyFrom(t, 0, 0, Ints.checkedCast(t.size()));
        return t2;
    }

    public Tokenizer getTokenizer(){
        return this.tokenizer;
    }

    public TensorCache getTensorCache(){
        return tensorCache;
    }


    protected  ClassifyOutput loadClassifierWeights(){
        throw new IllegalArgumentException("loadClassifierWeights not yet implemented");
    }

    protected PoolingLayer loadPoolingWeights() {
        return null;
    }

    public TokenRenderer getTokenRenderer(){
        return this.tokenRenderer;
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