package io.teknek.reckoning.model.granitemoehybrid;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.generator.EmbedInput;
import io.teknek.reckoning.generator2.CausalSelfAttention2;
import io.teknek.reckoning.generator2.FeedForward2;
import io.teknek.reckoning.generator2.LayerNorm2;
import io.teknek.reckoning.generator2.RmsNorm2;
import io.teknek.reckoning.generator2.SampleOutputRef;
import io.teknek.reckoning.generator2.TransformerBlock2;
import io.teknek.reckoning.grace.PreTrainedTokenizer;
import io.teknek.reckoning.math.WrappedForkJoinPool;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.tensorparallel.TensorParallelCollectives;
import io.teknek.reckoning.model.tensorparallel.TensorParallelContext;
import io.teknek.reckoning.safetensors.Config;
import io.teknek.reckoning.safetensors.WeightLoader;
import io.teknek.reckoning.tensor.KvBufferCacheSettings;
import io.teknek.reckoning.tensor2.TensorRef;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.toolcallparser.ToolCallParser;

import java.util.Optional;
import java.util.stream.IntStream;


public class GraniteMoeHybridModel extends AbstractModel {

    private volatile TensorRef embedTokenWeightsRef;

    public GraniteMoeHybridModel(InferenceType inferenceType, Config config, WeightLoader weightLoader,
            PreTrainedTokenizer tokenizer, DType workingMemoryDType, DType workingMemoryQType, Optional<DType> modelQType,
            MetricRegistry metricRegistry,
            KvBufferCacheSettings kvBufferCacheSettings, ToolCallParser toolCallParser,
            WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization, Lighter lighter) {
        super(inferenceType, config, weightLoader, tokenizer, workingMemoryDType, workingMemoryQType, modelQType,
                metricRegistry, kvBufferCacheSettings, toolCallParser, pool,
                tensorParallelContext, tensorParallelCollectives, outputHeadQuantization, lighter);
    }

    @Override
    protected EmbedInput loadInputWeights() {
        embedTokenWeightsRef = registerModelTensorRef(weights.loadRef("model.embed_tokens.weight"));
        return new EmbedInput(this) {
                @Override
                public TensorRef batchInputsToEmbeddings(int[] inputTokens, int startPos) {
                    TensorRef embeddings = makeDenseTensorRef(inputTokens.length, config.embeddingLength);
                    try {
                        for (int row = 0; row < inputTokens.length; row++) {
                            try (TensorRef source = embedTokenWeightsRef.slice(inputTokens[row])) {
                                if (source.dType() == workingDType) {
                                    lighter.copy(source, 0, embeddings, embeddings.shape().getOffset(row, 0),
                                            config.embeddingLength);
                                } else {
                                    try (TensorRef converted = lighter.reshape(source, workingDType)) {
                                        lighter.copy(converted, 0, embeddings, embeddings.shape().getOffset(row, 0),
                                                config.embeddingLength);
                                    }
                                }
                            }
                        }
                        scaleEmbeddings(embeddings);
                        return embeddings;
                    } catch (RuntimeException | Error e) {
                        embeddings.close();
                        throw e;
                    }
                }

                @Override
                public TensorRef inputTokenToEmbedding(int inputToken, int position) {
                    try (TensorRef source = embedTokenWeightsRef.slice(inputToken)) {
                        TensorRef embedding = makeDenseTensorRef(1, config.embeddingLength);
                        try {
                            if (source.dType() == workingDType) {
                                lighter.copy(source, 0, embedding, 0, config.embeddingLength);
                            } else {
                                try (TensorRef converted = lighter.reshape(source, workingDType)) {
                                    lighter.copy(converted, 0, embedding, 0, config.embeddingLength);
                                }
                            }
                            scaleEmbeddings(embedding);
                            return embedding;
                        } catch (RuntimeException | Error e) {
                            embedding.close();
                            throw e;
                        }
                    }
                }

                private void scaleEmbeddings(TensorRef embeddings) {
                    if (config.embeddingMultiplier != null) {
                        for (int row = 0; row < embeddings.shape().first(); row++) {
                            for (int column = 0; column < config.embeddingLength; column++) {
                                embeddings.set(embeddings.get(row, column) * config.embeddingMultiplier, row, column);
                            }
                        }
                    }
                }
        };
    }


    @Override
    protected SampleOutputRef loadOutputWeightsRef() {
        DType qType = modelQType.orElse(this.modelDType);
        TensorRef norm = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                "model.norm.weight", weights::loadRef, qType));
        String outputWeightName = weights.isWeightPresent("lm_head.weight")
                ? "lm_head.weight" : "model.embed_tokens.weight";
        TensorRef head = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                outputWeightName, weights::loadRef, outputHeadQuantization.orElse(workingDType)));
        return new SampleOutputRef() {
            @Override
            public LayerNorm2 outputLayerNorm() {
                return new RmsNorm2(GraniteMoeHybridModel.this, norm, 0.0f);
            }

            @Override
            public TensorRef outputLogitsWeights() {
                return head;
            }
        };
    }

    @Override
    protected boolean usesTensorRefExecution() {
        return true;
    }

    @Override
    public boolean usesKvCache2Generation() {
        return true;
    }

    @Override
    protected boolean addBosToken() {
        return false;
    }


    @Override
    protected TransformerBlock2[] loadTransformerBlockWeights2() {
        GraniteMoeHybridConfig graniteConfig = (GraniteMoeHybridConfig) this.config;
        DType qType = modelQType.orElse(this.modelDType);
        TransformerBlock2[] blocks = new TransformerBlock2[graniteConfig.numberOfLayers];
        IntStream.range(0, graniteConfig.numberOfLayers).parallel().forEach(i -> {
            String base = "model.layers." + i + ".";
            io.teknek.reckoning.generator2.SelfAttention2 attention;
            if ("mamba".equals(graniteConfig.layerTypes.get(i))) {
                String prefix = base + "mamba.";
                attention = new GraniteMoeHybridMambaLayer2(this, graniteConfig, i,
                        registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(prefix + "in_proj.weight", weights::loadRef, qType)),
                        registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(prefix + "conv1d.weight", weights::loadRef, qType)),
                        graniteConfig.mambaConvBias ? registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                                prefix + "conv1d.bias", weights::loadRef, qType)) : null,
                        registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(prefix + "dt_bias", weights::loadRef, qType)),
                        registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(prefix + "A_log", weights::loadRef, qType)),
                        registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(prefix + "D", weights::loadRef, qType)),
                        registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(prefix + "norm.weight", weights::loadRef, qType)),
                        registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(prefix + "out_proj.weight", weights::loadRef, qType)));
            } else if ("attention".equals(graniteConfig.layerTypes.get(i))) {
                String attentionPrefix = base + "self_attn.";
                String qName = attentionPrefix + "q_proj.weight";
                String kName = attentionPrefix + "k_proj.weight";
                String vName = attentionPrefix + "v_proj.weight";
                String oName = attentionPrefix + "o_proj.weight";
                attention = new CausalSelfAttention2(this, i,
                        registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(qName, weights::loadRef, qType)),
                        registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(kName, weights::loadRef, qType)),
                        registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(vName, weights::loadRef, qType)),
                        registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(oName, weights::loadRef, qType)),
                        lighter, metricRegistry, qName, kName, vName, oName);
            } else {
                throw new IllegalArgumentException("Unsupported GraniteMoeHybrid layer type: " + graniteConfig.layerTypes.get(i));
            }

            String inputName = base + "shared_mlp.input_linear.weight";
            String outputName = base + "shared_mlp.output_linear.weight";
            TensorRef inputRef = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                    inputName, weights::loadRef, qType));
            TensorRef outputRef = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                    outputName, weights::loadRef, qType));
            FeedForward2 feedForward = new GraniteMoeHybridSharedMlp2(this, inputRef, outputRef, i);
            if (graniteConfig.numLocalExperts > 0) {
                String moePrefix = base + "block_sparse_moe.";
                feedForward = new GraniteMoeHybridMoeFeedForward2(this, graniteConfig, feedForward,
                        registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                                moePrefix + "router.layer.weight", weights::loadRef, qType)),
                        registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                                moePrefix + "input_linear.weight", weights::loadRef, qType)),
                        registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                                moePrefix + "output_linear.weight", weights::loadRef, qType)));
            }

            String inputNormName = base + "input_layernorm.weight";
            String postAttentionNormName = base + "post_attention_layernorm.weight";
            blocks[i] = new TransformerBlock2(this, i,
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(
                            loadAndMaybeQuantizedExcluding1DTensors(inputNormName, weights::loadRef, qType)), 0.0f)),
                    attention,
                    Optional.empty(),
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(
                            loadAndMaybeQuantizedExcluding1DTensors(postAttentionNormName, weights::loadRef, qType)), 0.0f)),
                     feedForward, Optional.empty(), Optional.empty());
        });
        return blocks;
    }

}
