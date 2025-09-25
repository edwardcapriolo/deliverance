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
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.Q8ByteBufferTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.TensorOperationsProvider;
import io.teknek.deliverance.tokenizer.Tokenizer;
import jdk.incubator.vector.FloatVector;
import net.jafama.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

interface Generator extends Closeable {

}
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
    //protected EmbedInput embedInput;
    //protected SampleOutput sampleOutput;
    //protected ClassifyOutput classifyOutput;
    //protected Optional<PoolingLayer> poolingLayer;
    //protected TransformerBlock[] transformerBlocks;
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

        //this.embedInput = inferenceType.isInput ? loadInputWeights() : null;
        //this.transformerBlocks = inferenceType.isFwdPass ? loadTransformerBlockWeights() : null;
        //this.sampleOutput = inferenceType.isOutput ? loadOutputWeights() : null;
        //this.classifyOutput = inferenceType.isClassify ? loadClassifierWeights() : null;
        //this.poolingLayer = inferenceType.isPooling ? Optional.ofNullable(loadPoolingWeights()) : Optional.empty();
    }

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
}