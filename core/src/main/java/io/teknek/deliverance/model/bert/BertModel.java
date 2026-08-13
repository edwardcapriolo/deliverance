
package io.teknek.deliverance.model.bert;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.classifier.ClassifyOutput;
import io.teknek.deliverance.embedding.PoolingLayer;
import io.teknek.deliverance.embedding.PoolingType;
import io.teknek.deliverance.embedding.SentenceTransformersEmbeddingRecipe;
import io.teknek.deliverance.embedding.SentenceTransformersPooling;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.grace.EncodeOptions;
import io.teknek.deliverance.grace.Encoding;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensorlib.TensorPlan;
import io.teknek.deliverance.tensorlib.TensorRuntimeGlobal;
import io.teknek.deliverance.toolcallparser.ToolCallParser;

import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.Optional;

public class BertModel extends AbstractModel {

    private static final String[] prefixes = new String[] { "", "bert." };

    private AbstractTensor wordEmbeddings;
    private AbstractTensor tokenTypeEmbeddings;
    private AbstractTensor positionEmbeddings;
    private LayerNorm inputLayerNorm;
    private SentenceTransformersEmbeddingRecipe embeddingRecipe = SentenceTransformersEmbeddingRecipe.defaultMeanNormalize();

    public BertModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer tokenizer, DType workingDType, DType workingQType,
                        Optional<DType> modelQType, ConfigurableTensorProvider configurableTensorProvider,
                        MetricRegistry metricRegistry, TensorAllocator arrayQueueTensorAllocator, KvBufferCacheSettings kvBufferCacheSettings,
                        ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
                        TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization) {
        //note: jLAMA uses FOrward_passs
        super(inferenceType, c, w, tokenizer, workingDType, workingQType, modelQType,
                configurableTensorProvider, metricRegistry, arrayQueueTensorAllocator, kvBufferCacheSettings, toolCallParser, pool,
                tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
    }

    protected AbstractTensor loadWeight(String name) {
        for (String prefix : prefixes) {
            String key = prefix + name;
            if (weights.isWeightPresent(key)) {
                return weights.load(key);
            }
            String alias = bertWeightAlias(name);
            if (!alias.equals(name)) {
                String aliasKey = prefix + alias;
                if (weights.isWeightPresent(aliasKey)) {
                    return weights.load(aliasKey);
                }
            }
        }
        throw new NoSuchElementException(Arrays.toString(prefixes) + " " + name + " not found in weights " + weights.tensorInfoMap());
    }

    private static String bertWeightAlias(String name) {
        if (name.endsWith("LayerNorm.weight")) {
            return name.substring(0, name.length() - "weight".length()) + "gamma";
        }
        if (name.endsWith("LayerNorm.bias")) {
            return name.substring(0, name.length() - "bias".length()) + "beta";
        }
        return name;
    }

    @Override
    protected EmbedInput loadInputWeights() {
        wordEmbeddings = loadWeight("embeddings.word_embeddings.weight");
        tokenTypeEmbeddings = loadWeight("embeddings.token_type_embeddings.weight");
        positionEmbeddings = loadWeight("embeddings.position_embeddings.weight");
        embeddingRecipe = SentenceTransformersEmbeddingRecipe.fromModelRoot(weights.modelRoot());
        inputLayerNorm = new LayerNorm(this, loadWeight("embeddings.LayerNorm.bias"),
                loadWeight("embeddings.LayerNorm.weight"), new MetricRegistry(),
                "model.weights.embeddings.LayerNorm.bias", "model.weights.embeddings.LayerNorm.weight");

        return new EmbedInput(BertModel.this) {
            @Override
            public AbstractTensor inputTokenToEmbedding(int inputToken, int position) {
                return bertEmbeddings(new BertInput(new int[] { inputToken }, null, null,
                        new int[] { position }, 1, 1));
            }
        };
    }

    /**
     * HF {@code BertEmbeddings.forward} for {@code input_ids}, {@code token_type_ids}, and {@code position_ids}.
     * Dropout is intentionally omitted because Deliverance inference corresponds to HF {@code model.eval()}.
     */
    public AbstractTensor bertEmbeddings(BertInput input) {
        if (wordEmbeddings == null || tokenTypeEmbeddings == null || positionEmbeddings == null || inputLayerNorm == null) {
            throw new IllegalStateException("BertModel.init() must be called before bertEmbeddings");
        }
        AbstractTensor embedding = makeDenseTensor(input.flattenedLength(), config.embeddingLength);
        bertEmbeddingGatherAddPlan(input, embedding).materialize();
        AbstractTensor normalized = inputLayerNorm.forward(embedding);
        embedding.close();
        return normalized;
    }

    TensorPlan.Tensor bertEmbeddingGatherAddPlan(BertInput input, AbstractTensor embedding) {
        TensorPlan plan = new TensorPlan(configurableTensorProvider.get(), getPool(), metricRegistry,
                TensorRuntimeGlobal.get(metricRegistry, getTensorRuntimeMode(), getPool().getCoreCount()));
        return plan.fuseRowsIntStream("bert_embeddings.gather_add", embedding.shape())
                .read("word_embeddings", plan.immutable("model.weights.embeddings.word_embeddings.weight", wordEmbeddings))
                .read("token_type_embeddings", plan.immutable("model.weights.embeddings.token_type_embeddings.weight", tokenTypeEmbeddings))
                .read("position_embeddings", plan.immutable("model.weights.embeddings.position_embeddings.weight", positionEmbeddings))
                .write("embedding", plan.mutable("embedding", embedding))
                .map("embedding = word[input_ids] + token_type[token_type_ids] + position[position_ids]",
                        TensorPlan.TensorOp.CUSTOM, "bert_embeddings.gather_add",
                        (ctx, rowOffset, rowCount) -> configurableTensorProvider.get().gatherRowsAdd(
                                ctx.tensor("embedding"), ctx.tensor("word_embeddings"), input.inputIds(),
                                ctx.tensor("token_type_embeddings"), input.tokenTypeIds(),
                                ctx.tensor("position_embeddings"), input.positionIds(), (int) rowOffset,
                                (int) rowCount))
                .tensor()
                .as("bert_embeddings.output");
    }

    String bertEmbeddingPathPlan(BertInput input) {
        try (AbstractTensor rawEmbedding = makeDenseTensor(input.flattenedLength(), config.embeddingLength);
             AbstractTensor normalizedEmbedding = makeDenseTensor(input.flattenedLength(), config.embeddingLength)) {
            String gatherPlan = bertEmbeddingGatherAddPlan(input, rawEmbedding).plan();
            String layerNormPlan = inputLayerNorm.forwardPlan(rawEmbedding, normalizedEmbedding,
                    "bert_embeddings.layernorm", "bert_embeddings.output", 0, config.embeddingLength).plan();
            return "[1] Gather/Add\n" + gatherPlan + "\n[2] LayerNorm\n" + layerNormPlan;
        }
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        TransformerBlock[] transformerBlocks = new TransformerBlock[config.numberOfLayers];

        for (int i = 0; i < config.numberOfLayers; i++) {
            String b = "encoder.layer." + i + ".";
            String prefix = b + "attention.";

            AbstractTensor keyBias = loadWeight(prefix + "self.key.bias");
            AbstractTensor keyWeight = loadWeight(prefix + "self.key.weight");

            AbstractTensor queryBias = loadWeight(prefix + "self.query.bias");
            AbstractTensor queryWeight = loadWeight(prefix + "self.query.weight");

            AbstractTensor valueBias = loadWeight(prefix + "self.value.bias");
            AbstractTensor valueWeight = loadWeight(prefix + "self.value.weight");

            AbstractTensor outputBias = loadWeight(prefix + "output.dense.bias");
            AbstractTensor outputWeight = loadWeight(prefix + "output.dense.weight");
            BertSelfAttention attention = new BertSelfAttention(
                    this,
                    i,
                    Optional.of(queryBias),
                    Optional.of(keyBias),
                    Optional.of(valueBias),
                    queryWeight,
                    keyWeight,
                    valueWeight,
                    Optional.of(outputBias),
                    outputWeight,
                    this.configurableTensorProvider,
                    metricRegistry
            );

            prefix = b;
            MLPBlock mlpBlock = new MLPBlock(
                    this,
                    config.activationFunction,
                    loadWeight(prefix + "intermediate.dense.bias"),
                    loadWeight(prefix + "intermediate.dense.weight"),
                    loadWeight(prefix + "output.dense.bias"),
                    loadWeight(prefix + "output.dense.weight"),
                    configurableTensorProvider
            );

            LayerNorm postAttentionNorm = new LayerNorm(this,
                    loadWeight(b + "attention.output.LayerNorm.bias"), loadWeight(b + "attention.output.LayerNorm.weight"),
                    metricRegistry
            );
            LayerNorm postMlpNorm = new LayerNorm(this, loadWeight(b + "output.LayerNorm.bias"), loadWeight(b + "output.LayerNorm.weight"), metricRegistry);

            transformerBlocks[i] = new BertTransformerBlock(this, i, attention, postAttentionNorm, mlpBlock,
                    postMlpNorm, configurableTensorProvider);
        }

        return transformerBlocks;
    }

    public AbstractTensor batchForward(BertInput input, KvBufferCache.KvBuffer kvbuf) {
        AbstractTensor embedding = bertEmbeddings(input);
        for (int i = 0; i < config.numberOfLayers; i++) {
            AbstractTensor previous = embedding;
            if (transformerBlocks[i] instanceof BertTransformerBlock bertBlock) {
                embedding = bertBlock.forward(previous, 0, kvbuf, Optional.empty(), ForwardPhase.PREFILL,
                        input.batchSize(), input.sequenceLength(), input.attentionMask());
            } else {
                embedding = transformerBlocks[i].forward(previous, 0, kvbuf, Optional.empty(), ForwardPhase.PREFILL);
            }
            previous.close();
        }
        return embedding;
    }

    @Override
    protected float[] timedEmbedding(String input, PoolingType poolingType) {
        if (poolingType != PoolingType.AVG) {
            return super.timedEmbedding(input, poolingType);
        }
        Encoding encoding = tokenizer.encode(input, EncodeOptions.defaults());
        if (encoding.length() >= config.contextLength) {
            throw new IllegalArgumentException("Encoded input length " + encoding.length()
                    + " exceeds context length " + config.contextLength);
        }
        try (KvBufferCache.KvBuffer kvMem = kvBufferCache.getEphemeralKvBuffer();
             AbstractTensor tokenEmbeddings = batchForward(BertInput.singleSequence(encoding.inputIds(),
                     encoding.attentionMask(), null, null), kvMem)) {
            SentenceTransformersPooling.Mode[] modes = embeddingRecipe.poolingModes()
                    .toArray(SentenceTransformersPooling.Mode[]::new);
            float[] embedding = SentenceTransformersPooling.pool(tokenEmbeddings, encoding.attentionMask(), 1,
                    encoding.length(), modes)[0];
            if (embeddingRecipe.normalize()) {
                SentenceTransformersPooling.normalize(embedding);
            }
            return embedding;
        }
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        throw new UnsupportedOperationException();
    }

    @Override
    protected ClassifyOutput loadClassifierWeights() {
        if (config.isClassifier()) {
            final AbstractTensor classifierWeight = loadWeight("classifier.weight");
            final AbstractTensor classifierBias = loadWeight("classifier.bias");

            return new ClassifyOutput() {
                @Override
                public AbstractTensor getClassificationWeights() {
                    return classifierWeight;
                }

                @Override
                public Optional<AbstractTensor> getClassificationBias() {
                    return Optional.of(classifierBias);
                }
            };
        } else {
            throw new UnsupportedOperationException("Classification not supported by this model");
        }
    }

    @Override
    protected PoolingLayer loadPoolingWeights() {
        // Return null if pooler weights are not present, allowing AVG pooling to be used instead
        // This is needed for models like LEAF that don't have a pooler layer
        if (!weights.isWeightPresent("pooler.dense.weight") && !weights.isWeightPresent("bert.pooler.dense.weight")) {
            return null;
        }
        final AbstractTensor poolerDenseWeight = loadWeight("pooler.dense.weight");
        final AbstractTensor poolerDenseBias = loadWeight("pooler.dense.bias");
        return new PoolingLayer() {
            public AbstractTensor getPoolingWeights() {
                return poolerDenseWeight;
            }

            public Optional<AbstractTensor> getPoolingBias() {
                return Optional.of(poolerDenseBias);
            }
        };
    }

}
