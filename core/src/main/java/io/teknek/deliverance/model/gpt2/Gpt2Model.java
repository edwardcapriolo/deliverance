package io.teknek.deliverance.model.gpt2;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.TokenRenderer;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tokenizer.Tokenizer;
import io.teknek.deliverance.toolcallparser.ToolCallParser;

import java.util.Optional;

public class Gpt2Model extends AbstractModel{

    public Gpt2Model(AbstractModel.InferenceType inferenceType, Config c, WeightLoader w, Tokenizer t, DType workingMemoryDType,
                     DType workingMemoryQType, Optional<DType> modelQType,
                     ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry,
                     TensorAllocator arrayQueueTensorAllocator, KvBufferCacheSettings kvBufferCacheSettings, TokenRenderer tokenRenderer,
                     ToolCallParser toolCallParser, WrappedForkJoinPool pool) {
        super(inferenceType, c, w, t, workingMemoryDType, workingMemoryQType, modelQType, configurableTensorProvider,
                metricRegistry, arrayQueueTensorAllocator, kvBufferCacheSettings, tokenRenderer, toolCallParser, pool);
    }

    @Override
    protected EmbedInput loadInputWeights() {
        final AbstractTensor wte = weights.load("wte.weight");
        final AbstractTensor wpe = weights.load("wpe.weight");

        return new EmbedInput(this) {
            @Override
            public AbstractTensor inputTokenToEmbedding(int inputToken, int position) {
                AbstractTensor embedding = makeDenseTensor(1, config.embeddingLength);
                for (int i = 0; i < config.embeddingLength; i++) {
                    float v = wte.get(inputToken, i) + wpe.get(position, i);
                    embedding.set(v, 0, i);
                }
                return embedding;
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
        TransformerBlock[] transformerBlocks = new TransformerBlock[config.dctx().numberOfLayers];

        for (int i = config.dctx().layerStart; i < config.dctx().layerEnd; i++) {
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
}
