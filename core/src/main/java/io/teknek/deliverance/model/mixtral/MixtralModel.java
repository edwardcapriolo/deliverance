package io.teknek.deliverance.model.mixtral;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.model.llama.*;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.MixtureOfExpertsBlock;
import io.teknek.deliverance.model.TokenRenderer;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tokenizer.Tokenizer;
import io.teknek.deliverance.toolcallparser.ToolCallParser;

import java.util.Optional;
import java.util.stream.IntStream;

import static io.teknek.deliverance.tensor.AbstractTensorUtils.quantize;

public class MixtralModel extends LlamaModel {

    public MixtralModel(InferenceType inferenceType, Config c, WeightLoader w, Tokenizer t, DType workingMemoryDType,
                        DType workingMemoryQType, Optional<DType> modelQType,
                        ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry,
                        TensorCache tensorCache, KvBufferCacheSettings kvBufferCacheSettings, TokenRenderer tokenRenderer,
                        ToolCallParser toolCallParser) {
        super(inferenceType, c, w, t, workingMemoryDType, workingMemoryQType, modelQType, configurableTensorProvider,
                metricRegistry, tensorCache, kvBufferCacheSettings, tokenRenderer, toolCallParser);
    }

/*
    @Override
    protected EmbedInput loadInputWeights() {
        throw new UnsupportedOperationException("not needed");
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        throw new UnsupportedOperationException("not needed");
    } */

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        MixtralConfig mixtralConfig = (MixtralConfig) config;
        DType qType = modelQType.orElse(this.modelDType);
        TransformerBlock[] transformerBlocks = new TransformerBlock[config.dctx().numberOfLayers];

        IntStream.range(config.dctx().layerStart, config.dctx().layerEnd).parallel().forEach(i -> {
            String base = "model.layers." + i + ".";
            String prefix = base + "self_attn.";

            CausalSelfAttention attention = new CausalSelfAttention(
                    this,
                    i,
                    quantize(weights.load(prefix + "q_proj.weight", config.dctx(), true, false), qType),
                    quantize(weights.load(prefix + "k_proj.weight", config.dctx(), true, false), qType),
                    quantize(weights.load(prefix + "v_proj.weight", config.dctx(), true, false), qType),
                    quantize(weights.load(prefix + "o_proj.weight"), qType),
                    this.configurableTensorProvider,
                    this.metricRegistry
            );

            prefix = base + "block_sparse_moe.";

            AbstractTensor[] expertGateWeights = new AbstractTensor[mixtralConfig.numberOfExperts];
            AbstractTensor[] expertDownWeights = new AbstractTensor[mixtralConfig.numberOfExperts];
            AbstractTensor[] expertUpWeights = new AbstractTensor[mixtralConfig.numberOfExperts];

            for (int e = 0; e < mixtralConfig.numberOfExperts; e++) {
                String expertPrefix = prefix + "experts." + e + ".";
                quantize(expertGateWeights[e] = weights.load(expertPrefix + "w1.weight", config.dctx(), true, false), qType);
                quantize(expertDownWeights[e] = weights.load(expertPrefix + "w2.weight"), qType);
                expertUpWeights[e] = quantize(weights.load(expertPrefix + "w3.weight", config.dctx(), true, false), qType);
            }

            MixtureOfExpertsBlock moe = new MixtureOfExpertsBlock(
                    this,
                    mixtralConfig.numberOfExperts,
                    mixtralConfig.numberOfExpertsPerToken,
                    config.activationFunction,
                    quantize(weights.load(prefix + "gate.weight"), qType),
                    expertGateWeights, // w1
                    expertDownWeights, // w2
                    expertUpWeights
            ); // w3

            transformerBlocks[i] = new TransformerBlock(
                    this,
                    i,
                    new RmsNorm(this, quantize(weights.load(base + "input_layernorm.weight"), qType), metricRegistry),
                    attention,
                    new RmsNorm(this, quantize(weights.load(base + "post_attention_layernorm.weight"), qType), metricRegistry),
                    moe,
                    configurableTensorProvider);

        });
        return transformerBlocks;
    }
}
