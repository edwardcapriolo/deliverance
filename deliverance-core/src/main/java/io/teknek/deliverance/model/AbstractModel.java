package io.teknek.deliverance.model;



import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;

import java.io.Closeable;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.*;
import io.teknek.deliverance.tensor.operations.TensorOperationsProvider;
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
    protected final Config c;
    protected final WeightLoader weights;
    protected final Tokenizer tokenizer;
    protected final DType modelDType;
    protected final DType workingDType;
    protected final DType workingQType;
    protected final Optional<DType> modelQType;
    protected EmbedInput embedInput;
    protected SampleOutput sampleOutput;
    //protected ClassifyOutput classifyOutput;
    //protected Optional<PoolingLayer> poolingLayer;
    protected TransformerBlock[] transformerBlocks;
    protected KvBufferCache kvBufferCache;


    protected AbstractModel(
            InferenceType inferenceType,
            Config c,
            WeightLoader w,
            Tokenizer t,
            DType workingMemoryDType,
            DType workingMemoryQType,
            Optional<DType> modelQType
    ) {
        this.inferenceType = inferenceType;
        this.c = c;
        this.weights = w;
        this.tokenizer = t;

        this.modelDType = w.getModelDType();
        this.workingDType = workingMemoryDType;
        this.modelQType = modelQType;
        this.kvBufferCache = new KvBufferCache(this);

        if (workingMemoryQType == null) {
            workingMemoryQType = TensorOperationsProvider.get().preferredWorkingQuantizedType();
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
        if (modelDType == DType.Q4 && workingMemoryQType.size() < TensorOperationsProvider.get().preferredWorkingQuantizedType().size()) {
            workingMemoryQType = TensorOperationsProvider.get().preferredWorkingQuantizedType();
        }

        if (workingMemoryQType != workingMemoryDType) {
            boolean supportsQType;
            AbstractTensor tmp = makeDenseTensor(Q8ByteBufferTensor.BLOCK_SIZE);
            try (AbstractTensor tmp2 = TensorOperationsProvider.get().quantize(tmp, workingMemoryQType, 0, Q8ByteBufferTensor.BLOCK_SIZE)) {
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

        logger.info(
                "Model type = {}, Working memory type = {}, Quantized memory type = {}",
                this.modelDType,
                this.workingDType,
                this.workingQType
        );

        this.embedInput = inferenceType.isInput ? loadInputWeights() : null;
        this.transformerBlocks = inferenceType.isFwdPass ? loadTransformerBlockWeights() : null;
        this.sampleOutput = inferenceType.isOutput ? loadOutputWeights() : null;
        //this.classifyOutput = inferenceType.isClassify ? loadClassifierWeights() : null;
        //this.poolingLayer = inferenceType.isPooling ? Optional.ofNullable(loadPoolingWeights()) : Optional.empty();
    }

    protected abstract EmbedInput loadInputWeights();
    protected abstract SampleOutput loadOutputWeights();
    protected abstract  TransformerBlock[] loadTransformerBlockWeights();

    @Override
    public void close() {
        kvBufferCache.close();
    }

    public Config getConfig(){
        return c;
    }

    public AbstractTensor makeTensor(int... shape) {
        TensorShape s = TensorShape.of(shape);
        return c.tensorCache.get(workingDType, s);
    }

    public AbstractTensor makeDenseTensor(int... shape) {
        return c.tensorCache.get(workingDType, TensorShape.of(shape));
    }

    public AbstractTensor makeDenseTensor(TensorShape s) {
        return c.tensorCache.get(workingDType, s);
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

    public Response generate(UUID sessionId, PromptContext promptContext, float temperature, int ntokens,
                      BiConsumer<String, Float> onTokenWithTimings) {
        long[] encoded = tokenizer.encode(promptContext.getPrompt());
        if (encoded.length > 0 && encoded[0] == c.bosToken) {
            encoded = Arrays.copyOfRange(encoded, 1, encoded.length);
        }
        Preconditions.checkArgument(encoded.length < c.contextLength && encoded.length < ntokens, "Prompt exceeds max tokens");


        try (KvBufferCache.KvBuffer kvmem = kvBufferCache.getKvBuffer(sessionId)) { // k and v for context window
            int startPos = kvmem.getCurrentContextPosition(); // Number of tokens in the buffer
            if (ntokens > c.contextLength) {
                ntokens = c.contextLength;
            }
            FinishReason reason = FinishReason.MAX_TOKENS;
            int promptLength;
            long promptBatchTime;
            int tokensGenerated;
            StringBuilder responseText = new StringBuilder();
            StringBuilder responseTextWithSpecialTokens = new StringBuilder();

            try (AbstractTensor logits = makeDenseTensor(c.vocabularySize)) {
                int[] promptTokens;

                if (addBosToken()) {
                    promptTokens = new int[(1 + encoded.length)];
                    promptTokens[0] = c.bosToken;
                    for (int i = 1; i <= encoded.length; i++)
                        promptTokens[i] = Ints.checkedCast(encoded[i - 1]);
                } else {
                    promptTokens = Arrays.stream(encoded).mapToInt(Ints::checkedCast).toArray();
                }
                promptLength = encoded.length;
                long start = System.currentTimeMillis();
                long promptStart = start;
                AbstractTensor last = batchForward(promptTokens, startPos, kvmem);
                promptBatchTime = System.currentTimeMillis() - start;
                float batchMsPerToken = Math.round((((double) promptBatchTime) / (double) promptLength));
                int next = sample(last.slice(last.shape().first() - 1), temperature, ThreadLocalRandom.current().nextFloat(), logits);
                float genMsPerToken = 0;
                tokensGenerated = 0;
                last.close();
                try {
                    String c = tokenizer.decode(next);
                    if (tokenizer.getModel().isSpecialToken(next)) {
                        responseTextWithSpecialTokens.append(c);
                    } else {
                        onTokenWithTimings.accept(c, batchMsPerToken);
                        responseText.append(c);
                        responseTextWithSpecialTokens.append(c);
                    }
                } catch (Exception e) {
                    logger.error("Failed to decode token {}", next, e);
                }

                start = System.currentTimeMillis();
                for (int i = startPos + promptTokens.length; i < ntokens; i++) {
                    AbstractTensor output = forward(next, i, kvmem);
                    tokensGenerated++;

                    next = sample(output, temperature, ThreadLocalRandom.current().nextFloat(), logits);

                    if (logger.isTraceEnabled()) logger.trace("Sampled token {} with temperature {}", next, temperature);
                    output.close();

                    kvmem.incrementContextPosition();

                    // Model may tell us it's done
                    if (c.eosTokens.contains(next)) {
                        reason = FinishReason.STOP_TOKEN;
                        break;
                    }

                    try {
                        String c = tokenizer.decode(next);

                        if (tokenizer.getModel().isSpecialToken(next)) {
                            responseTextWithSpecialTokens.append(c);
                        } else {
                            genMsPerToken = (System.currentTimeMillis() - start) / (float) (tokensGenerated);
                            onTokenWithTimings.accept(c, genMsPerToken);
                            responseTextWithSpecialTokens.append(c);
                            responseText.append(c);
                        }
                    } catch (Exception e) {
                        logger.error("Failed to decode token {}", next, e);
                    }
                }

                long end = System.currentTimeMillis();

                Response response = new Response(
                        responseText.toString(),
                        responseTextWithSpecialTokens.toString(),
                        reason,
                        promptLength,
                        tokensGenerated,
                        promptBatchTime,
                        end - start
                );

                return response;
            }


        }

    }

    public int sample(AbstractTensor output, float temperature, float uniformSample, AbstractTensor logits) {
        try (AbstractTensor embedding = sampleOutput.getOutputLayerNorm().forward(output)) {
            // This is a mix of argmax and sampling with softmax
            VectorMath.pchunk(0, c.vocabularySize, (chunkStart, chunkSize) -> {
                TensorOperationsProvider.get()
                        .dotProductChunk(logits, embedding, sampleOutput.getOutputLogitsWeights(), 0, c.embeddingLength, chunkStart, chunkSize);
            });

            if (c.logitMultiplier != null) {
                TensorOperationsProvider.get().scale(1.0f / c.logitMultiplier, logits, 0, c.vocabularySize);
            }

            int maxi = Integer.MIN_VALUE;
            double maxv = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < c.vocabularySize; i++) {
                float v = logits.get(0, i);
                if (c.finalLogitSoftCapping != null) {
                    v /= c.finalLogitSoftCapping;
                    v = (float) FastMath.tanh(v);
                    v = v * c.finalLogitSoftCapping;
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
            for (int i = 0; i < c.vocabularySize; i++) {
                float v = (float) FastMath.exp((logits.get(0, i) - maxv) / temperature);
                sum += v;
                logits.set(v, 0, i);
            }

            float acc = 0;
            for (int i = 0; i < c.vocabularySize; i++) {
                float v = logits.get(0, i) / sum;
                acc += v;
                if (acc >= uniformSample) return i;
            }

            return c.vocabularySize - 1;
        }
    }


    public AbstractTensor batchForward(int[] token_ids, int startPos, KvBufferCache.KvBuffer kvbuf) {
        return batchForward(token_ids, startPos, kvbuf, Optional.empty());
    }

    public AbstractTensor batchForward(int[] token_ids, int startPos, KvBufferCache.KvBuffer kvbuf,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        AbstractTensor embedding = null;
        for (int i = 0; i < token_ids.length; i += MAX_BATCH_SIZE) {
            int[] batch = Arrays.copyOfRange(token_ids, i, Math.min(token_ids.length, i + MAX_BATCH_SIZE));
            embedding = embedInput.batchInputsToEmbeddings(batch, startPos + i);
            embedding = forward(embedding, startPos + i, kvbuf, tensorReducer);
            logger.debug("Batched forward pass for tokens {} to {}", i, i + batch.length);
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
        for (int i = c.dctx().layerStart; i < c.dctx().layerEnd; i++) {
            int relativeLayer = i - c.dctx().layerStart;
            AbstractTensor ref = embedding; // reference so we can free
            embedding = transformerBlocks[relativeLayer].forward(embedding, startPos, kvbuf, tensorReducer);
            ref.close();
        }

        return embedding;
    }

    /** I do not follow how this is supposed to maybeQuantize since we always take the type from t? */
    public AbstractTensor maybeQuantize(AbstractTensor t) {
        AbstractTensor t2 = c.tensorCache.get(t.dType(), t.shape());
        t2.copyFrom(t, 0, 0, Ints.checkedCast(t.size()));
        return t2;
    }

    public Tokenizer getTokenizer(){
        return this.tokenizer;
    }
}