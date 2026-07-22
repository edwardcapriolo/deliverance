package io.teknek.deliverance.model.granitemoehybrid;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.EmbedInput;
import io.teknek.deliverance.generator.FeedForward;
import io.teknek.deliverance.generator.LayerNorm;
import io.teknek.deliverance.generator.RmsNorm;
import io.teknek.deliverance.generator.SampleOutput;
import io.teknek.deliverance.generator.SelfAttention;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.toolcallparser.ToolCallParser;

import java.util.Optional;
import java.util.stream.IntStream;

import static io.teknek.deliverance.tensor.AbstractTensorUtils.quantize;

public class GraniteMoeHybridModel extends AbstractModel {

    private volatile AbstractTensor embedTokenWeights;

    public GraniteMoeHybridModel(InferenceType inferenceType, Config config, WeightLoader weightLoader,
            PreTrainedTokenizer tokenizer, DType workingMemoryDType, DType workingMemoryQType, Optional<DType> modelQType,
            ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry,
            TensorAllocator tensorAllocator, KvBufferCacheSettings kvBufferCacheSettings, ToolCallParser toolCallParser,
            WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization) {
        super(inferenceType, config, weightLoader, tokenizer, workingMemoryDType, workingMemoryQType, modelQType,
                configurableTensorProvider, metricRegistry, tensorAllocator, kvBufferCacheSettings, toolCallParser, pool,
                tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
    }

    @Override
    protected EmbedInput loadInputWeights() {
        if (this.embedTokenWeights == null) {
            this.embedTokenWeights = quantize(this.weights.load("model.embed_tokens.weight"), this.workingDType);
            this.configurableTensorProvider.get().registerModelTensor(this.embedTokenWeights);
        }
        return new EmbedInput(this) {
            @Override
            public AbstractTensor inputTokenToEmbedding(int inputToken, int position) {
                AbstractTensor tokenEmbedding = GraniteMoeHybridModel.this.embedTokenWeights.slice(true, inputToken);
                AbstractTensor source = tokenEmbedding;
                if (tokenEmbedding.dType() != GraniteMoeHybridModel.this.workingDType) {
                    // Embedding scaling happens on the working tensor. Some providers do not
                    // support scaling BF16 rows directly, and the rest of the forward path
                    // expects working dtype activations anyway.
                    source = GraniteMoeHybridModel.this.configurableTensorProvider.get()
                            .quantize(tokenEmbedding, GraniteMoeHybridModel.this.workingDType, 0,
                                    parent.getConfig().embeddingLength);
                }
                AbstractTensor embedding = parent.getTensorAllocator()
                        .getDirty(GraniteMoeHybridModel.this.workingDType, source.shape());
                embedding.copyFrom(source, 0, 0, parent.getConfig().embeddingLength);
                if (source != tokenEmbedding) {
                    source.close();
                }
                if (parent.getConfig().embeddingMultiplier != null) {
                    GraniteMoeHybridModel.this.configurableTensorProvider.get().scale(parent.getConfig().embeddingMultiplier,
                            embedding, 0, parent.getConfig().embeddingLength);
                }
                return embedding;
            }
        };
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        DType qType = this.modelQType.orElse(this.modelDType);
        LayerNorm outputLayerNorm = new RmsNorm(this, quantize(this.weights.load("model.norm.weight"), qType),
                this.metricRegistry);
        DType outputHeadDType = this.outputHeadQuantization.orElse(this.workingDType);
        boolean forceOutputHeadQuantization = this.outputHeadQuantization.isPresent();
        String outputWeightName = this.weights.isWeightPresent("lm_head.weight")
                ? "lm_head.weight"
                : "model.embed_tokens.weight";
        AbstractTensor outputWeights = quantize(this.weights.load(outputWeightName), outputHeadDType,
                forceOutputHeadQuantization);
        this.configurableTensorProvider.get().registerModelTensor(outputWeights);
        return new SampleOutput() {
            @Override
            public LayerNorm getOutputLayerNorm() {
                return outputLayerNorm;
            }

            @Override
            public AbstractTensor getOutputLogitsWeights() {
                return outputWeights;
            }
        };
    }

    @Override
    protected boolean addBosToken() {
        return false;
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        GraniteMoeHybridConfig graniteConfig = (GraniteMoeHybridConfig) this.config;
        DType qType = this.modelQType.orElse(this.modelDType);
        TransformerBlock[] blocks = new TransformerBlock[graniteConfig.numberOfLayers];
        IntStream.range(0, graniteConfig.numberOfLayers).parallel().forEach(i -> {
            String base = "model.layers." + i + ".";
            SelfAttention attention;
            if ("mamba".equals(graniteConfig.layerTypes.get(i))) {
                String mambaPrefix = base + "mamba.";
                attention = new GraniteMoeHybridMambaLayer(
                        this,
                        graniteConfig,
                        quantize(this.weights.load(mambaPrefix + "in_proj.weight"), qType),
                        quantize(this.weights.load(mambaPrefix + "conv1d.weight"), qType),
                        graniteConfig.mambaConvBias
                                ? Optional.of(quantize(this.weights.load(mambaPrefix + "conv1d.bias"), qType))
                                : Optional.empty(),
                        quantize(this.weights.load(mambaPrefix + "dt_bias"), qType),
                        quantize(this.weights.load(mambaPrefix + "A_log"), qType),
                        quantize(this.weights.load(mambaPrefix + "D"), qType),
                        quantize(this.weights.load(mambaPrefix + "norm.weight"), qType),
                        quantize(this.weights.load(mambaPrefix + "out_proj.weight"), qType),
                        this.configurableTensorProvider
                );
            } else if ("attention".equals(graniteConfig.layerTypes.get(i))) {
                String attentionPrefix = base + "self_attn.";
                attention = new GraniteMoeHybridAttention(
                        this,
                        i,
                        quantize(this.weights.load(attentionPrefix + "q_proj.weight"), qType),
                        quantize(this.weights.load(attentionPrefix + "k_proj.weight"), qType),
                        quantize(this.weights.load(attentionPrefix + "v_proj.weight"), qType),
                        quantize(this.weights.load(attentionPrefix + "o_proj.weight"), qType),
                        this.configurableTensorProvider,
                        this.metricRegistry
                );
            } else {
                throw new IllegalArgumentException("Unsupported GraniteMoeHybrid layer type: " + graniteConfig.layerTypes.get(i));
            }

            FeedForward sharedMlp = new GraniteMoeHybridSharedMlp(
                    this,
                    graniteConfig,
                    quantize(this.weights.load(base + "shared_mlp.input_linear.weight"), qType),
                    quantize(this.weights.load(base + "shared_mlp.output_linear.weight"), qType),
                    this.configurableTensorProvider
            );
            FeedForward feedForward = sharedMlp;
            if (graniteConfig.numLocalExperts > 0) {
                String moePrefix = base + "block_sparse_moe.";
                feedForward = new GraniteMoeHybridMoeFeedForward(
                        this,
                        graniteConfig,
                        sharedMlp,
                        quantize(this.weights.load(moePrefix + "router.layer.weight"), qType),
                        quantize(this.weights.load(moePrefix + "input_linear.weight"), qType),
                        quantize(this.weights.load(moePrefix + "output_linear.weight"), qType),
                        this.configurableTensorProvider
                );
            }

            blocks[i] = new TransformerBlock(
                    this,
                    i,
                    Optional.of(new RmsNorm(this, quantize(this.weights.load(base + "input_layernorm.weight"), qType),
                            this.metricRegistry)),
                    attention,
                    Optional.empty(),
                    Optional.of(new RmsNorm(this, quantize(this.weights.load(base + "post_attention_layernorm.weight"), qType),
                            this.metricRegistry)),
                    feedForward,
                    Optional.empty(),
                    Optional.empty(),
                    this.configurableTensorProvider
            );
        });
        return blocks;
    }

}
