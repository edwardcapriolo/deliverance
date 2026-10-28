package io.teknek.deliverance.model.bert;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.embedding.SentenceTransformersEmbeddingRecipe;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.generator2.BertMLPBlock2;
import io.teknek.deliverance.generator2.BertSelfAttention2;
import io.teknek.deliverance.generator2.BertTransformerBlock2;
import io.teknek.deliverance.generator2.LayerNorm2;
import io.teknek.deliverance.generator2.TransformerBlock2;
import io.teknek.deliverance.grace.Encoding;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensorlib.TensorPlan;
import io.teknek.deliverance.toolcallparser.ToolCallParser;

import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.Optional;

/** Shared encoder, embedding, and tensor execution for BERT-family models. */
public abstract class BertFamilyModel extends AbstractModel {
    protected AbstractTensor wordEmbeddings;
    protected AbstractTensor tokenTypeEmbeddings;
    protected AbstractTensor positionEmbeddings;
    protected LayerNorm inputLayerNorm;
    protected TensorRef wordEmbeddingsRef;
    protected TensorRef tokenTypeEmbeddingsRef;
    protected TensorRef positionEmbeddingsRef;
    protected TensorRef inputLayerNormBiasRef;
    protected TensorRef inputLayerNormWeightRef;
    protected SentenceTransformersEmbeddingRecipe embeddingRecipe = SentenceTransformersEmbeddingRecipe.defaultMeanNormalize();

    protected BertFamilyModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer tokenizer,
            DType workingDType, DType workingQType, Optional<DType> modelQType,
            ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry,
            TensorAllocator tensorAllocator, KvBufferCacheSettings kvBufferCacheSettings,
            ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization) {
        super(inferenceType, c, w, tokenizer, workingDType, workingQType, modelQType,
                configurableTensorProvider, metricRegistry, tensorAllocator, kvBufferCacheSettings, toolCallParser,
                pool, tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
    }

    @Override
    protected boolean usesTensorRefExecution() {
        return true;
    }

    @Override
    public boolean usesKvCache2Generation() {
        return true;
    }

    protected AbstractTensor loadWeight(String name) {
        for (String prefix : weightPrefixes()) {
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
        throw new NoSuchElementException(Arrays.toString(weightPrefixes()) + " " + name
                + " not found in weights " + weights.tensorInfoMap());
    }

    protected String[] weightPrefixes() {
        return new String[] { "", "bert." };
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

    protected TensorRef loadWeightRef(String name) {
        for (String prefix : weightPrefixes()) {
            String key = prefix + name;
            if (weights.isWeightPresent(key)) {
                return weights.loadRef(key);
            }
            String alias = bertWeightAlias(name);
            if (!alias.equals(name)) {
                String aliasKey = prefix + alias;
                if (weights.isWeightPresent(aliasKey)) {
                    return weights.loadRef(aliasKey);
                }
            }
        }
        throw new NoSuchElementException(Arrays.toString(weightPrefixes()) + " " + name
                + " not found in weights " + weights.tensorInfoMap());
    }

    protected BertInput inputForEncoding(Encoding encoding) {
        return BertInput.singleSequence(encoding.inputIds(), encoding.attentionMask(), null, null);
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

        wordEmbeddingsRef = registerModelTensorRef(loadWeightRef("embeddings.word_embeddings.weight"));
        tokenTypeEmbeddingsRef = registerModelTensorRef(loadWeightRef("embeddings.token_type_embeddings.weight"));
        positionEmbeddingsRef = registerModelTensorRef(loadWeightRef("embeddings.position_embeddings.weight"));
        inputLayerNormBiasRef = registerModelTensorRef(loadWeightRef("embeddings.LayerNorm.bias"));
        inputLayerNormWeightRef = registerModelTensorRef(loadWeightRef("embeddings.LayerNorm.weight"));

        return new EmbedInput(BertFamilyModel.this) {
            @Override
            public TensorRef inputTokenToEmbedding(int inputToken, int position) {
                return bertEmbeddingsRef(new BertInput(new int[] { inputToken }, null, null,
                        new int[] { position }, 1, 1));
            }
        };
    }

    @Override
    protected TransformerBlock2[] loadTransformerBlockWeights2() {
        DType qType = modelQType.orElse(modelDType);
        TransformerBlock2[] blocks = new TransformerBlock2[config.numberOfLayers];
        for (int i = 0; i < config.numberOfLayers; i++) {
            String base = "encoder.layer." + i + ".";
            String attention = base + "attention.";
            TensorRef queryWeight = loadRef2(attention + "self.query.weight", qType);
            TensorRef keyWeight = loadRef2(attention + "self.key.weight", qType);
            TensorRef valueWeight = loadRef2(attention + "self.value.weight", qType);
            TensorRef outputWeight = loadRef2(attention + "output.dense.weight", qType);
            TensorRef queryBias = registerModelTensorRef(loadWeightRef(attention + "self.query.bias"));
            TensorRef keyBias = registerModelTensorRef(loadWeightRef(attention + "self.key.bias"));
            TensorRef valueBias = registerModelTensorRef(loadWeightRef(attention + "self.value.bias"));
            TensorRef outputBias = registerModelTensorRef(loadWeightRef(attention + "output.dense.bias"));
            BertSelfAttention2 selfAttention = new BertSelfAttention2(this, queryWeight, keyWeight, valueWeight,
                    outputWeight, queryBias, keyBias, valueBias, outputBias, lighter);

            TensorRef attentionNormBias = registerModelTensorRef(loadWeightRef(base + "attention.output.LayerNorm.bias"));
            TensorRef attentionNormWeight = registerModelTensorRef(loadWeightRef(base + "attention.output.LayerNorm.weight"));
            TensorRef intermediateWeight = loadRef2(base + "intermediate.dense.weight", qType);
            TensorRef intermediateBias = registerModelTensorRef(loadWeightRef(base + "intermediate.dense.bias"));
            TensorRef outputMlpWeight = loadRef2(base + "output.dense.weight", qType);
            TensorRef outputMlpBias = registerModelTensorRef(loadWeightRef(base + "output.dense.bias"));
            BertMLPBlock2 mlp = new BertMLPBlock2(this, intermediateWeight, outputMlpWeight,
                    intermediateBias, outputMlpBias, lighter);
            TensorRef outputNormBias = registerModelTensorRef(loadWeightRef(base + "output.LayerNorm.bias"));
            TensorRef outputNormWeight = registerModelTensorRef(loadWeightRef(base + "output.LayerNorm.weight"));
            blocks[i] = new BertTransformerBlock2(this, i, selfAttention,
                    new LayerNorm2(this, attentionNormBias, attentionNormWeight, metricRegistry), mlp,
                    new LayerNorm2(this, outputNormBias, outputNormWeight, metricRegistry));
        }
        return blocks;
    }

    private TensorRef loadRef2(String name, DType qType) {
        return registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(name, this::loadWeightRef, qType));
    }

    protected TensorRef bertEmbeddingsRef(BertInput input) {
        TensorRef embedding = makeDenseTensorRef(input.flattenedLength(), config.embeddingLength);
        for (int row = 0; row < input.flattenedLength(); row++) {
            int token = input.inputIds()[row];
            int tokenType = input.tokenTypeIds()[row];
            int position = input.positionIds()[row];
            for (int column = 0; column < config.embeddingLength; column++) {
                embedding.set(wordEmbeddingsRef.get(token, column)
                        + tokenTypeEmbeddingsRef.get(tokenType, column)
                        + positionEmbeddingsRef.get(position, column), row, column);
            }
        }
        TensorRef normalized = new LayerNorm2(this, inputLayerNormBiasRef, inputLayerNormWeightRef,
                metricRegistry).forward(embedding);
        embedding.close();
        return normalized;
    }

    public AbstractTensor bertEmbeddings(BertInput input) {
        if (wordEmbeddings == null || tokenTypeEmbeddings == null || positionEmbeddings == null || inputLayerNorm == null) {
            throw new IllegalStateException("BertFamilyModel.init() must be called before bertEmbeddings");
        }
        AbstractTensor embedding = makeDenseTensor(input.flattenedLength(), config.embeddingLength);
        bertEmbeddingGatherAddPlan(input, embedding).materialize();
        AbstractTensor normalized = inputLayerNorm.forward(embedding);
        embedding.close();
        return normalized;
    }

    protected TensorPlan.Tensor bertEmbeddingGatherAddPlan(BertInput input, AbstractTensor embedding) {
        TensorPlan plan = new TensorPlan(configurableTensorProvider.get(), getPool(), metricRegistry,
                getTensorRuntime());
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
            BertSelfAttention attention = new BertSelfAttention(this, i, Optional.of(queryBias), Optional.of(keyBias),
                    Optional.of(valueBias), queryWeight, keyWeight, valueWeight, Optional.of(outputBias), outputWeight,
                    configurableTensorProvider, metricRegistry);
            prefix = b;
            MLPBlock mlpBlock = new MLPBlock(this, config.activationFunction,
                    loadWeight(prefix + "intermediate.dense.bias"), loadWeight(prefix + "intermediate.dense.weight"),
                    loadWeight(prefix + "output.dense.bias"), loadWeight(prefix + "output.dense.weight"), configurableTensorProvider);
            LayerNorm postAttentionNorm = new LayerNorm(this, loadWeight(b + "attention.output.LayerNorm.bias"),
                    loadWeight(b + "attention.output.LayerNorm.weight"), metricRegistry);
            LayerNorm postMlpNorm = new LayerNorm(this, loadWeight(b + "output.LayerNorm.bias"),
                    loadWeight(b + "output.LayerNorm.weight"), metricRegistry);
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
}
