package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;

import java.util.*;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

import io.teknek.deliverance.CausualWhisperer;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.embedding.PoolingLayer;
import io.teknek.deliverance.embedding.PoolingType;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.math.VectorMathUtils;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.*;
import io.teknek.deliverance.tensor.impl.Q8ByteBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tokenizer.Tokenizer;
import jdk.incubator.vector.FloatVector;
import net.jafama.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static io.teknek.deliverance.tensor.DebugSupport.debug;


public abstract class AbstractModel implements Generator {
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


    protected AbstractModel(InferenceType inferenceType, Config c, WeightLoader w, Tokenizer t, DType workingMemoryDType,
                            DType workingMemoryQType, Optional<DType> modelQType, ConfigurableTensorProvider provider,
                            MetricRegistry metricRegistry, TensorCache tensorCache, KvBufferCacheSettings kvBufferCacheSettings) {
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
        logger.info("Model type = {}, Working memory type = {}, Quantized memory type = {}", modelDType, workingDType,
                workingQType);
        this.embedInput = inferenceType.isInput ? loadInputWeights() : null;
        this.transformerBlocks = inferenceType.isFwdPass ? loadTransformerBlockWeights() : null;
        this.sampleOutput = inferenceType.isOutput ? loadOutputWeights() : null;

        //embedding
        this.poolingLayer = inferenceType.isPooling ? Optional.ofNullable(loadPoolingWeights()) : Optional.empty();
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

    public Optional<PromptSupport> promptSupport() {
        return tokenizer.promptSupport();
    }

    protected boolean addBosToken() {
        return true;
    }

    public Response generate(UUID sessionId, PromptContext promptContext, GeneratorParameters generatorParameters,
                      BiConsumer<String, Float> onTokenWithTimings) {
        Random random = generatorParameters.seed.isEmpty() ? new Random(): new Random(generatorParameters.seed.get());
        long[] encoded = tokenizer.encode(promptContext.getPrompt());
        if (encoded.length > 0 && encoded[0] == config.bosToken) {
            encoded = Arrays.copyOfRange(encoded, 1, encoded.length);
        }
        int ntokens = generatorParameters.ntokens.orElse(256);
        float temperature = generatorParameters.temperature.orElse(0.0f);
        String cacheSalt = generatorParameters.cacheSalt.orElse("sha1here");
        Preconditions.checkArgument(encoded.length < config.contextLength
                && encoded.length < ntokens, "Prompt exceeds max tokens");
        try (KvBufferCache.KvBuffer kvmem = kvBufferCache.getKvBuffer(cacheSalt)) { // k and v for context window
            int startPos = kvmem.getCurrentContextPosition(); // Number of tokens in the buffer
            if (ntokens > config.contextLength) {
                ntokens = config.contextLength;
            }
            FinishReason reason = FinishReason.MAX_TOKENS;
            int promptLength;
            long promptBatchTime;
            int tokensGenerated;
            StringBuilder responseText = new StringBuilder();
            StringBuilder responseTextWithSpecialTokens = new StringBuilder();

            try (AbstractTensor logits = makeDenseTensor(config.vocabularySize)) {
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
                logger.warn("promptTokens: {}", promptTokens);
                promptLength = encoded.length;
                long start = System.currentTimeMillis();
                AbstractTensor last = batchForward(promptTokens, startPos, kvmem);
                logger.warn("After batch forward size: {} shape: {}" , last.size(), last.shape());
                promptBatchTime = System.currentTimeMillis() - start;
                float batchMsPerToken = Math.round((((double) promptBatchTime) / (double) promptLength));
                int next = sample(last.slice(last.shape().first() - 1), temperature, random.nextFloat(), logits);
                float genMsPerToken = 0;
                tokensGenerated = 0;
                last.close();
                String decoded = tokenizer.decode(next);
                String cleaned = tokenizer.tokenForResponse(decoded);
                if (tokenizer.getModel().isSpecialToken(next)) {
                    responseTextWithSpecialTokens.append(cleaned);
                } else {
                    onTokenWithTimings.accept(cleaned, batchMsPerToken);
                    responseText.append(cleaned);
                    responseTextWithSpecialTokens.append(cleaned);
                }
                start = System.currentTimeMillis();
                for (int i = startPos + promptTokens.length; i < ntokens; i++) {
                    AbstractTensor output = forward(next, i, kvmem);
                    tokensGenerated++;
                    next = sample(output, temperature, random.nextFloat(), logits);
                    if (logger.isTraceEnabled()) {
                        logger.trace("Sampled token {} with temperature {}", next, temperature);
                    }
                    output.close();
                    kvmem.incrementContextPosition();
                    if (config.eosTokens.contains(next)) {
                        reason = FinishReason.STOP_TOKEN;
                        break;
                    }
                    String decoded1 = tokenizer.decode(next);
                    CausualWhisperer.LOGGER.debug("decoded for response {}", decoded1);
                    String cleaned2 = tokenizer.tokenForResponse(decoded1);
                    if (tokenizer.getModel().isSpecialToken(next)) {
                        responseTextWithSpecialTokens.append(cleaned2);
                    } else {
                        genMsPerToken = (System.currentTimeMillis() - start) / (float) (tokensGenerated);
                        onTokenWithTimings.accept(cleaned2, genMsPerToken);
                        responseTextWithSpecialTokens.append(cleaned2);
                        responseText.append(cleaned2);
                    }
                }

                long end = System.currentTimeMillis();
                //post process response is still missing
                return new Response(responseText.toString(), responseTextWithSpecialTokens.toString(),
                        reason, promptLength, tokensGenerated, promptBatchTime, end - start);
            }
        }
    }

    public float[] embed(String input, PoolingType poolingType) {
        CausualWhisperer.LOGGER.info("embedding on {} using pooling type {}", input, poolingType);
        int[] encoded = Arrays.stream(tokenizer.encode(input)).mapToInt(Ints::checkedCast).toArray();
        Preconditions.checkArgument(encoded.length < config.contextLength);
        float [] outputEmbedding = new float[config.embeddingLength];
        CausualWhisperer.LOGGER.info("created float [] outputEmbedding of length {}", config.embeddingLength);

        try (KvBufferCache.KvBuffer kvMem = kvBufferCache.getEphemeralKvBuffer()){
            int promptLength = encoded.length;
            float avgp = 1.0f / promptLength;
            CausualWhisperer.LOGGER.info("1.0f / promptLength {} = avgp {}", promptLength, avgp);

            try (AbstractTensor r = batchForward(encoded, 0, kvMem)){
                if (poolingType == PoolingType.MODEL){
                    if (poolingLayer.isEmpty()){
                        throw new UnsupportedOperationException("no pooling layer for this model");
                    }
                    AbstractTensor output = r.slice(promptLength - 1);
                    AbstractTensor pooled = makeDenseTensor(1, config.embeddingLength);
                    configurableTensorProvider.get()
                            .batchDotProduct(pooled, output, poolingLayer.get().getPoolingWeights(), 0, 0, config.embeddingLength);

                    configurableTensorProvider.get()
                            .batchDotProduct(pooled, output, poolingLayer.get().getPoolingWeights(), 0, 0, config.embeddingLength);

                    VectorMath.pfor(0, config.embeddingLength, i -> {
                        // BERT seems to use tanh for pooling rather than gelu

                        //outputEmbedding[i] = ActivationFunction.eval(ActivationFunction.Type.TANH, pooled.get(0, i));
                        outputEmbedding[i] = ActivationFunction.eval(config.activationFunction, pooled.get(0, i));
                    });
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

    public int sample(AbstractTensor output, float temperature, float uniformSample, AbstractTensor logits) {
        try (AbstractTensor embedding = sampleOutput.getOutputLayerNorm().forward(output)) {
            // This is a mix of argmax and sampling with softmax
            VectorMath.pchunk(0, config.vocabularySize, (chunkStart, chunkSize) -> {
                configurableTensorProvider.get()
                        .dotProductChunk(logits, embedding, sampleOutput.getOutputLogitsWeights(), 0, config.embeddingLength, chunkStart, chunkSize);
            }, configurableTensorProvider.get().parallelSplitSize());

            if (config.logitMultiplier != null) {
                configurableTensorProvider.get().scale(1.0f / config.logitMultiplier, logits, 0, config.vocabularySize);
            }

            int maxi = Integer.MIN_VALUE;
            double maxv = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < config.vocabularySize; i++) {
                float v = logits.get(0, i);
                if (config.finalLogitSoftCapping != null) {
                    v /= config.finalLogitSoftCapping;
                    v = (float) FastMath.tanh(v);
                    v = v * config.finalLogitSoftCapping;
                    logits.set(v, 0, i);
                }
                if (v > maxv) {
                    maxi = i;
                    maxv = v;
                }
            }

            if (temperature == 0.0) {
                return maxi;
            }

            float sum = 0;
            for (int i = 0; i < config.vocabularySize; i++) {
                float v = (float) FastMath.exp((logits.get(0, i) - maxv) / temperature);
                sum += v;
                logits.set(v, 0, i);
            }

            float acc = 0;
            for (int i = 0; i < config.vocabularySize; i++) {
                float v = logits.get(0, i) / sum;
                acc += v;
                if (acc >= uniformSample) return i;
            }

            return config.vocabularySize - 1;
        }
    }

    public AbstractTensor batchForward(int[] token_ids, int startPos, KvBufferCache.KvBuffer kvbuf) {
        return batchForward(token_ids, startPos, kvbuf, Optional.empty());
    }

    public AbstractTensor batchForward(int[] token_ids, int startPos, KvBufferCache.KvBuffer kvbuf,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        AbstractTensor embedding = null;
        CausualWhisperer.LOGGER.info("batchForward from 0 to token_ids.length {} max_batch_size {} per iteration",
                token_ids.length, MAX_BATCH_SIZE);
        for (int i = 0; i < token_ids.length; i += MAX_BATCH_SIZE) {
            int[] batch = Arrays.copyOfRange(token_ids, i, Math.min(token_ids.length, i + MAX_BATCH_SIZE));
            //logger.warn("batch forward i: {} batch: {}", i, batch);
            embedding = embedInput.batchInputsToEmbeddings(batch, startPos + i);
            //logger.warn("embedding {} {}", embedding.shape(), embedding.size());
            embedding = forward(embedding, startPos + i, kvbuf, tensorReducer);
            //logger.debug("Batched forward pass for tokens {} to {}", i, i + batch.length);
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
        debug("EMBEDDING TOKEN", token_id);
        debug("TOKEN POSITION", pos);
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


    protected PoolingLayer loadPoolingWeights() {
        return null;
    }
}