package io.teknek.deliverance.model.gpt2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.generator2.BertMLPBlock2;
import io.teknek.deliverance.generator2.Gpt2CausalSelfAttention2;
import io.teknek.deliverance.generator2.LayerNorm2;
import io.teknek.deliverance.generator2.SampleOutputRef;
import io.teknek.deliverance.generator2.TransformerBlock2;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor2.Split;
import io.teknek.deliverance.tensor2.Transpose;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.toolcallparser.ToolCallParser;

import java.util.Optional;

public class Gpt2Model extends AbstractModel{
    private volatile TensorRef wordEmbeddingsRef;

    public Gpt2Model(AbstractModel.InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer t, DType workingMemoryDType,
                      DType workingMemoryQType, Optional<DType> modelQType,
                      ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry,
                      TensorAllocator arrayQueueTensorAllocator, KvBufferCacheSettings kvBufferCacheSettings,
                      ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
                      TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization) {
        super(inferenceType, c, w, t, workingMemoryDType, workingMemoryQType, modelQType, configurableTensorProvider,
                metricRegistry, arrayQueueTensorAllocator, kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext,
                tensorParallelCollectives, outputHeadQuantization);
    }

    @Override
    protected EmbedInput loadInputWeights() {
        if (usesTensorRefExecution()) {
            wordEmbeddingsRef = registerModelTensorRef(weights.loadRef("wte.weight"));
            TensorRef positionEmbeddings = registerModelTensorRef(weights.loadRef("wpe.weight"));
            return new EmbedInput(this) {
                @Override
                public TensorRef inputTokenToEmbedding(int inputToken, int position) {
                    TensorRef embedding = makeDenseTensorRef(1, config.embeddingLength);
                    for (int i = 0; i < config.embeddingLength; i++) {
                        embedding.set(wordEmbeddingsRef.get(inputToken, i) + positionEmbeddings.get(position, i), 0, i);
                    }
                    return embedding;
                }
            };
        }
        final AbstractTensor wte = weights.load("wte.weight");
        final AbstractTensor wpe = weights.load("wpe.weight");

        return new EmbedInput(this) {
            @Override
            public TensorRef inputTokenToEmbedding(int inputToken, int position) {
                AbstractTensor embedding = makeDenseTensor(1, config.embeddingLength);
                for (int i = 0; i < config.embeddingLength; i++) {
                    float v = wte.get(inputToken, i) + wpe.get(position, i);
                    embedding.set(v, 0, i);
                }
                return TensorRef.owned(embedding);
            }
        };
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        final AbstractTensor wte = weights.load("wte.weight");
        this.configurableTensorProvider.get().registerModelTensor(wte);
        final LayerNorm layerNorm = new LayerNorm(this, weights.load("ln_f.bias"),
                weights.load("ln_f.weight"), metricRegistry);

        return new SampleOutput() {
            @Override
            public LayerNorm getOutputLayerNorm() {
                return layerNorm;
            }

            @Override
            public AbstractTensor getOutputLogitsWeights() {
                return wte;
            }
        };
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        TransformerBlock[] transformerBlocks = new TransformerBlock[config.numberOfLayers];

        for (int i = 0; i < config.numberOfLayers; i++) {
            String b = "h." + i + ".";
            String prefix = b + "attn.";

            AbstractTensor[] attnBias = weights.load(prefix + "c_attn.bias").split(3, 1);
            AbstractTensor[] attnWeights = weights.load(prefix + "c_attn.weight").transpose().split(3, 0);
            CausalSelfAttention attention = new CausalSelfAttention(
                    this,
                    i,
                    Optional.of(attnBias[0]),
                    Optional.of(attnBias[1]),
                    Optional.of(attnBias[2]),
                    attnWeights[0],
                    attnWeights[1],
                    attnWeights[2],
                    Optional.of(weights.load(prefix + "c_proj.bias")),
                    weights.load(prefix + "c_proj.weight").transpose(),
                    configurableTensorProvider,
                    metricRegistry
            );

            prefix = b + "mlp.";
            MLPBlock mlpBlock = new MLPBlock(
                    this,
                    config.activationFunction,
                    weights.load(prefix + "c_fc.bias"),
                    weights.load(prefix + "c_fc.weight").transpose(),
                    weights.load(prefix + "c_proj.bias"),
                    weights.load(prefix + "c_proj.weight").transpose(),
                    configurableTensorProvider
            );

            LayerNorm layerNorm1 = new LayerNorm(this, weights.load(b + "ln_1.bias"),
                    weights.load(b + "ln_1.weight"), metricRegistry);
            LayerNorm layerNorm2 = new LayerNorm(this, weights.load(b + "ln_2.bias"),
                    weights.load(b + "ln_2.weight"), metricRegistry);

            transformerBlocks[i] = new TransformerBlock(this, i, layerNorm1, attention, layerNorm2, mlpBlock,
                    configurableTensorProvider);
        }

        return transformerBlocks;
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
    protected TransformerBlock2[] loadTransformerBlockWeights2() {
        TransformerBlock2[] blocks = new TransformerBlock2[config.numberOfLayers];
        for (int i = 0; i < config.numberOfLayers; i++) {
            String base = "h." + i + ".";
            String attentionBase = base + "attn.";
            TensorRef[] qkvWeights = split(transpose(weights.loadRef(attentionBase + "c_attn.weight")), 3, 0);
            TensorRef[] qkvBiases = split(weights.loadRef(attentionBase + "c_attn.bias"), 3, 1);
            TensorRef outputWeight = transpose(weights.loadRef(attentionBase + "c_proj.weight"));
            TensorRef outputBias = registerModelTensorRef(weights.loadRef(attentionBase + "c_proj.bias"));
            Gpt2CausalSelfAttention2 attention = new Gpt2CausalSelfAttention2(this, i,
                    registerModelTensorRef(qkvWeights[0]), registerModelTensorRef(qkvWeights[1]),
                    registerModelTensorRef(qkvWeights[2]), registerModelTensorRef(outputWeight),
                    registerModelTensorRef(qkvBiases[0]), registerModelTensorRef(qkvBiases[1]),
                    registerModelTensorRef(qkvBiases[2]), outputBias, lighter, metricRegistry);

            String mlpBase = base + "mlp.";
            BertMLPBlock2 mlp = new BertMLPBlock2(this,
                    registerModelTensorRef(transpose(weights.loadRef(mlpBase + "c_fc.weight"))),
                    registerModelTensorRef(transpose(weights.loadRef(mlpBase + "c_proj.weight"))),
                    registerModelTensorRef(weights.loadRef(mlpBase + "c_fc.bias")),
                    registerModelTensorRef(weights.loadRef(mlpBase + "c_proj.bias")), lighter);
            blocks[i] = new TransformerBlock2(this, i,
                    Optional.of(new LayerNorm2(this, registerModelTensorRef(weights.loadRef(base + "ln_1.bias")),
                            registerModelTensorRef(weights.loadRef(base + "ln_1.weight")), metricRegistry)),
                    attention, Optional.empty(),
                    Optional.of(new LayerNorm2(this, registerModelTensorRef(weights.loadRef(base + "ln_2.bias")),
                            registerModelTensorRef(weights.loadRef(base + "ln_2.weight")), metricRegistry)),
                    mlp, Optional.empty(), Optional.empty(), configurableTensorProvider);
        }
        return blocks;
    }

    @Override
    protected SampleOutputRef loadOutputWeightsRef() {
        TensorRef finalNormBias = registerModelTensorRef(weights.loadRef("ln_f.bias"));
        TensorRef finalNormWeight = registerModelTensorRef(weights.loadRef("ln_f.weight"));
        return new SampleOutputRef() {
            @Override
            public LayerNorm2 outputLayerNorm() {
                return new LayerNorm2(Gpt2Model.this, finalNormBias, finalNormWeight, metricRegistry);
            }

            @Override
            public TensorRef outputLogitsWeights() {
                return wordEmbeddingsRef;
            }
        };
    }

    private TensorRef transpose(TensorRef source) {
        TensorRef destination = lighter.allocate(source.dType(),
                TensorShape.of(source.shape().last(), source.shape().first()));
        try {
            lighter.transpose(new Transpose(source).destination(destination));
            return destination;
        } finally {
            source.close();
        }
    }

    private TensorRef[] split(TensorRef source, int chunks, int dimension) {
        TensorRef[] destinations = Split.allocate(lighter, source, chunks, dimension);
        try {
            lighter.split(new Split().source(source).into(destinations).chunks(chunks).dimension(dimension));
            return destinations;
        } finally {
            source.close();
        }
    }
}
