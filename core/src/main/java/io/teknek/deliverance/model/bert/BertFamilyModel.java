package io.teknek.deliverance.model.bert;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.embedding.SentenceTransformersEmbeddingRecipe;
import io.teknek.deliverance.generator.EmbedInput;
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
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.toolcallparser.ToolCallParser;

import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.Optional;

/** Shared encoder, embedding, and tensor execution for BERT-family models. */
public abstract class BertFamilyModel extends AbstractModel {
    protected TensorRef wordEmbeddingsRef;
    protected TensorRef tokenTypeEmbeddingsRef;
    protected TensorRef positionEmbeddingsRef;
    protected TensorRef inputLayerNormBiasRef;
    protected TensorRef inputLayerNormWeightRef;
    protected SentenceTransformersEmbeddingRecipe embeddingRecipe = SentenceTransformersEmbeddingRecipe.defaultMeanNormalize();

    protected BertFamilyModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer tokenizer,
            DType workingDType, DType workingQType, Optional<DType> modelQType,
            MetricRegistry metricRegistry, KvBufferCacheSettings kvBufferCacheSettings,
            ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization, Lighter lighter) {
        super(inferenceType, c, w, tokenizer, workingDType, workingQType, modelQType,
                metricRegistry, kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext,
                tensorParallelCollectives, outputHeadQuantization, lighter);
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
        embeddingRecipe = SentenceTransformersEmbeddingRecipe.fromModelRoot(weights.modelRoot());

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

}
