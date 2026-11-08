package io.teknek.reckoning.model.gpt2;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.generator.*;
import io.teknek.reckoning.generator2.BertMLPBlock2;
import io.teknek.reckoning.generator2.Gpt2CausalSelfAttention2;
import io.teknek.reckoning.generator2.LayerNorm2;
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
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.tensor2.TensorRef;
import io.teknek.reckoning.tensor2.Split;
import io.teknek.reckoning.tensor2.Transpose;
import io.teknek.reckoning.tensor.TensorShape;
import io.teknek.reckoning.toolcallparser.ToolCallParser;

import java.util.Optional;

public class Gpt2Model extends AbstractModel{
    private volatile TensorRef wordEmbeddingsRef;

    public Gpt2Model(AbstractModel.InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer t, DType workingMemoryDType,
                      DType workingMemoryQType, Optional<DType> modelQType,
                       MetricRegistry metricRegistry, KvBufferCacheSettings kvBufferCacheSettings,
                       ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
                       TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization,
                       Lighter lighter) {
         super(inferenceType, c, w, t, workingMemoryDType, workingMemoryQType, modelQType,
                 metricRegistry, kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext,
                 tensorParallelCollectives, outputHeadQuantization, lighter);
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
        throw new IllegalStateException("GPT2 requires TensorRef execution");
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
                     mlp, Optional.empty(), Optional.empty());
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
