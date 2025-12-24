
package io.teknek.deliverance.model.bert;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.CausualWhisperer;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.embedding.PoolingLayer;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tokenizer.Tokenizer;

import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.Optional;

public class BertModel extends AbstractModel {

    private static final String[] prefixes = new String[] { "", "bert." };

    public BertModel(InferenceType inferenceType, Config c, WeightLoader w, Tokenizer tokenizer, DType workingDType, DType workingQType,
                     Optional<DType> modelQType, ConfigurableTensorProvider configurableTensorProvider,
                     MetricRegistry metricRegistry, TensorCache tensorCache, KvBufferCacheSettings kvBufferCacheSettings) {
        //note: jLAMA uses FOrward_passs
        super(inferenceType, c, w, tokenizer, workingDType, workingQType, modelQType,
                configurableTensorProvider, metricRegistry, tensorCache, kvBufferCacheSettings);
    }

    protected AbstractTensor loadWeight(String name) {
        for (String prefix : prefixes) {
            String key = prefix + name;
            if (weights.isWeightPresent(key)) {
                return weights.load(key);
            }
        }
        throw new NoSuchElementException(Arrays.toString(prefixes) + " " + name + " not found in weights "+weights.tensorInfoMap());
    }

    @Override
    protected EmbedInput loadInputWeights() {
        AbstractTensor we = loadWeight("embeddings.word_embeddings.weight");
        AbstractTensor wte = loadWeight("embeddings.token_type_embeddings.weight");
        AbstractTensor wpe = loadWeight("embeddings.position_embeddings.weight");


        LayerNorm inputLayerNorm = new LayerNorm(this, loadWeight("embeddings.LayerNorm.bias"),
                loadWeight("embeddings.LayerNorm.weight"), new MetricRegistry());

        return new EmbedInput(BertModel.this) {
            @Override
            public AbstractTensor inputTokenToEmbedding(int inputToken, int position) {
                AbstractTensor embedding = makeDenseTensor(config.embeddingLength);
                if (position == 3){
                    CausualWhisperer.LOGGER.info("BertModel.inputTokenToEmbedding {}", embedding.shape());
                    CausualWhisperer.LOGGER.info("BertModel.inputTokenToEmbedding inputToken: {} position: {} ", inputToken, position);
                }
                for (int i = 0; i < config.embeddingLength; i++) {
                    float v = we.get(inputToken, i) + wte.get(0, i) + wpe.get(position, i);
                    if (position==3 && i < 5 && CausualWhisperer.LOGGER.isInfoEnabled()){
                        CausualWhisperer.LOGGER.info( "inputTokenToEmbedding[{}] = word_embed_weight {} + type_embed_weight {} + position_embed_weight {} = v {}",
                        i, we.get(inputToken, i), wte.get(0, i), wpe.get(position, i), v);
                    }
                    embedding.set(v, 0, i);
                }
                AbstractTensor lnemb = inputLayerNorm.forward(embedding);
                embedding.close();
                return lnemb;
            }
        };
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        TransformerBlock[] transformerBlocks = new TransformerBlock[config.dctx().embeddingSegmentLength];

        for (int i = config.dctx().layerStart; i < config.dctx().layerEnd; i++) {
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
            CausalSelfAttention attention = new CausalSelfAttention(
                    this,
                    i,
                    Optional.of(keyBias),
                    Optional.of(queryBias),
                    Optional.of(valueBias),
                    keyWeight,
                    queryWeight,
                    valueWeight,
                    Optional.of(outputBias),
                    outputWeight,
                    this.configurableTensorProvider
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

            transformerBlocks[i] = new TransformerBlock(this, i, attention, postAttentionNorm, mlpBlock, postMlpNorm, configurableTensorProvider);
        }

        return transformerBlocks;
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        throw new UnsupportedOperationException();
    }

    @Override
    protected PoolingLayer loadPoolingWeights() {
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

    /*
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
    }*/
}