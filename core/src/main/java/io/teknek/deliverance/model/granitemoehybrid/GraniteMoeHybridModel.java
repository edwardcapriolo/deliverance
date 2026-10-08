package io.teknek.deliverance.model.granitemoehybrid;

import io.dropwizard.metrics5.MetricRegistry;
import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.EmbedInput;
import io.teknek.deliverance.generator.FeedForward;
import io.teknek.deliverance.generator.LayerNorm;
import io.teknek.deliverance.generator.RmsNorm;
import io.teknek.deliverance.generator.SampleOutput;
import io.teknek.deliverance.generator.SelfAttention;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.generator2.CausalSelfAttention2;
import io.teknek.deliverance.generator2.LayerNorm2;
import io.teknek.deliverance.generator2.RmsNorm2;
import io.teknek.deliverance.generator2.SampleOutputRef;
import io.teknek.deliverance.generator2.TransformerBlock2;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor2.TensorRef;
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
            // Keep quantized embeddings compressed. TensorRef generation only reads token rows;
            // expanding the entire vocabulary matrix here makes Q4 model startup unnecessarily expensive.
            AbstractTensor loaded = this.weights.load("model.embed_tokens.weight");
            this.embedTokenWeights = ((GraniteMoeHybridConfig) this.config).denseAttentionOnly()
                    ? loaded : quantize(loaded, this.workingDType);
            this.configurableTensorProvider.get().registerModelTensor(this.embedTokenWeights);
        }
        return new EmbedInput(this) {
            @Override
            public TensorRef batchInputsToEmbeddings(int[] inputTokens, int startPos) {
                try (Timer.Context ignored = InferenceProfiler.timer(parent.getMetricRegistry(), "embedinput.batch_inputs").time()) {
                    int hidden = parent.getConfig().embeddingLength;
                    AbstractTensor embeddings = parent.getTensorAllocator()
                            .getDirty(GraniteMoeHybridModel.this.workingDType, TensorShape.of(inputTokens.length, hidden));
                    for (int i = 0; i < inputTokens.length; i++) {
                        AbstractTensor tokenEmbedding = GraniteMoeHybridModel.this.embedTokenWeights.slice(true, inputTokens[i]);
                        AbstractTensor source = tokenEmbedding;
                        if (tokenEmbedding.dType() != GraniteMoeHybridModel.this.workingDType) {
                            source = GraniteMoeHybridModel.this.configurableTensorProvider.get()
                                    .quantize(tokenEmbedding, GraniteMoeHybridModel.this.workingDType, 0, hidden);
                        }
                        embeddings.copyFrom(source, 0, embeddings.getOffset(i, 0), hidden);
                        if (source != tokenEmbedding) {
                            source.close();
                        }
                    }
                    if (parent.getConfig().embeddingMultiplier != null) {
                        GraniteMoeHybridModel.this.scale(parent.getConfig().embeddingMultiplier, embeddings, 0, hidden);
                    }
                    return TensorRef.owned(embeddings);
                }
            }

            @Override
            public TensorRef inputTokenToEmbedding(int inputToken, int position) {
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
                    GraniteMoeHybridModel.this.scale(parent.getConfig().embeddingMultiplier,
                            embedding, 0, parent.getConfig().embeddingLength);
                }
                return TensorRef.owned(embedding);
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
        return ((GraniteMoeHybridConfig) config).denseAttentionOnly();
    }

    @Override
    public boolean usesKvCache2Generation() {
        return ((GraniteMoeHybridConfig) config).denseAttentionOnly();
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
                    this.configurableTensorProvider, i
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

    @Override
    protected TransformerBlock2[] loadTransformerBlockWeights2() {
        GraniteMoeHybridConfig graniteConfig = (GraniteMoeHybridConfig) this.config;
        if (!graniteConfig.denseAttentionOnly()) {
            throw new UnsupportedOperationException("TensorRef Granite execution requires dense attention-only layers");
        }
        DType qType = modelQType.orElse(this.modelDType);
        TransformerBlock2[] blocks = new TransformerBlock2[graniteConfig.numberOfLayers];
        IntStream.range(0, graniteConfig.numberOfLayers).parallel().forEach(i -> {
            String base = "model.layers." + i + ".";
            String attentionPrefix = base + "self_attn.";
            String qName = attentionPrefix + "q_proj.weight";
            String kName = attentionPrefix + "k_proj.weight";
            String vName = attentionPrefix + "v_proj.weight";
            String oName = attentionPrefix + "o_proj.weight";
            CausalSelfAttention2 attention = new CausalSelfAttention2(this, i,
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(qName, weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(kName, weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(vName, weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(oName, weights::loadRef, qType)),
                    lighter, metricRegistry, qName, kName, vName, oName);

            String inputName = base + "shared_mlp.input_linear.weight";
            String outputName = base + "shared_mlp.output_linear.weight";
            TensorRef inputRef = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                    inputName, weights::loadRef, qType));
            TensorRef outputRef = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                    outputName, weights::loadRef, qType));
            GraniteMoeHybridSharedMlp2 feedForward = new GraniteMoeHybridSharedMlp2(this, inputRef, outputRef, i);

            String inputNormName = base + "input_layernorm.weight";
            String postAttentionNormName = base + "post_attention_layernorm.weight";
            blocks[i] = new TransformerBlock2(this, i,
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(
                            loadAndMaybeQuantizedExcluding1DTensors(inputNormName, weights::loadRef, qType)), 0.0f)),
                    attention,
                    Optional.empty(),
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(
                            loadAndMaybeQuantizedExcluding1DTensors(postAttentionNormName, weights::loadRef, qType)), 0.0f)),
                    feedForward, Optional.empty(), Optional.empty(),
                    configurableTensorProvider);
        });
        return blocks;
    }

}
