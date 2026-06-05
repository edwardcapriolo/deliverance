package io.teknek.deliverance.model.mixtral;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.llama.*;
import io.teknek.deliverance.model.MixtureOfExpertsBlock;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.toolcallparser.ToolCallParser;

import java.util.Optional;
import java.util.stream.IntStream;

import static io.teknek.deliverance.tensor.AbstractTensorUtils.quantize;

public class MixtralModel extends LlamaModel {

    public MixtralModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer t, DType workingMemoryDType,
                        DType workingMemoryQType, Optional<DType> modelQType,
                        ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry,
                        TensorAllocator arrayQueueTensorAllocator, KvBufferCacheSettings kvBufferCacheSettings,
                        ToolCallParser toolCallParser, WrappedForkJoinPool pool) {
        super(inferenceType, c, w, t, workingMemoryDType, workingMemoryQType, modelQType, configurableTensorProvider,
                metricRegistry, arrayQueueTensorAllocator, kvBufferCacheSettings, toolCallParser, pool);
    }
    
    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        MixtralConfig mixtralConfig = (MixtralConfig) config;
        DType qType = modelQType.orElse(this.modelDType);
        TransformerBlock[] transformerBlocks = new TransformerBlock[config.numberOfLayers];

        IntStream.range(0, config.numberOfLayers).parallel().forEach(i -> {
            String base = "model.layers." + i + ".";
            String prefix = base + "self_attn.";

            CausalSelfAttention attention = new CausalSelfAttention(
                    this,
                    i,
                    quantize(weights.load(prefix + "q_proj.weight"), qType),
                    quantize(weights.load(prefix + "k_proj.weight"), qType),
                    quantize(weights.load(prefix + "v_proj.weight"), qType),
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
                expertGateWeights[e] = quantize(weights.load(expertPrefix + "w1.weight"), qType);
                expertDownWeights[e] = quantize(weights.load(expertPrefix + "w2.weight"), qType);
                expertUpWeights[e] = quantize(weights.load(expertPrefix + "w3.weight"), qType);
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
