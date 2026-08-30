package io.teknek.deliverance.model.qwen3;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.MLPBlock;
import io.teknek.deliverance.generator.Qwen3KvCacheSelfAttention;
import io.teknek.deliverance.generator.RmsNorm;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;

import static io.teknek.deliverance.tensor.AbstractTensorUtils.quantize;

public class Qwen3MoeModel extends Qwen3Model {
    private static final Logger LOGGER = LoggerFactory.getLogger(Qwen3MoeModel.class);

    public Qwen3MoeModel(InferenceType inferenceType, Config config, WeightLoader weights,
            PreTrainedTokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType,
            ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry,
            TensorAllocator arrayQueueTensorAllocator, KvBufferCacheSettings kvBufferCacheSettings,
            ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType,
                configurableTensorProvider, metricRegistry, arrayQueueTensorAllocator, kvBufferCacheSettings,
                toolCallParser, pool, tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
    }

    /**
     * Not supported: dense layers use {@code MLPBlock} (free LoRA coverage) but sparse layers route
     * through {@code Qwen3MoeFeedForward} (one base tensor name per expert, not per layer) -- a real
     * adapter targeting gate/up/down would apply silently only to dense layers, a confusing
     * partial result. Overrides {@code Qwen3Model}'s (inherited from {@code LlamaModel}) {@code
     * true} back to {@code false}. See step 4 plan Section 6.
     */
    @Override
    protected boolean supportsLoraHotSwap() {
        return false;
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        Qwen3MoeConfig moeConfig = (Qwen3MoeConfig) config;
        DType qType = modelQType.orElse(this.modelDType);
        TransformerBlock[] blocks = new TransformerBlock[config.numberOfLayers];
        LOGGER.info("loading qwen3_moe transformer blocks layers={} experts={} experts_per_token={} sparse_step={} mlp_only_layers={}",
                config.numberOfLayers, moeConfig.numExperts, moeConfig.numExpertsPerToken,
                moeConfig.decoderSparseStep, moeConfig.mlpOnlyLayers);
        for (int i = 0; i < config.numberOfLayers; i++) {
            boolean sparseLayer = moeConfig.sparseLayer(i);
            LOGGER.info("loading qwen3_moe layer={} sparse={} start", i, sparseLayer);
            String base = "model.layers." + i + ".";
            String attn = base + "self_attn.";
            LOGGER.debug("loading qwen3_moe layer={} attention weights prefix={}", i, attn);
            Qwen3KvCacheSelfAttention attention = new Qwen3KvCacheSelfAttention(
                    this,
                    i,
                    quantize(weights.load(attn + "q_proj.weight"), qType),
                    quantize(weights.load(attn + "k_proj.weight"), qType),
                    quantize(weights.load(attn + "v_proj.weight"), qType),
                    quantize(weights.load(attn + "o_proj.weight"), qType),
                    quantize(weights.load(attn + "q_norm.weight"), qType),
                    quantize(weights.load(attn + "k_norm.weight"), qType),
                    configurableTensorProvider,
                    metricRegistry,
                    null, null, null, null
            );
            LOGGER.debug("loaded qwen3_moe layer={} attention weights", i);

            String mlpPrefix = base + "mlp.";
            var mlp = sparseLayer
                    ? new Qwen3MoeFeedForward(this, moeConfig,
                            quantize(weights.load(mlpPrefix + "gate.weight"), qType),
                            expertWeights(i, mlpPrefix, "gate_proj.weight", moeConfig.numExperts, qType),
                            expertWeights(i, mlpPrefix, "up_proj.weight", moeConfig.numExperts, qType),
                            expertWeights(i, mlpPrefix, "down_proj.weight", moeConfig.numExperts, qType),
                            configurableTensorProvider,
                            Qwen3MoeFeedForward.PROVIDER_EXECUTION)
                    : new MLPBlock(
                            this,
                            config.activationFunction,
                            quantize(weights.load(mlpPrefix + "gate_proj.weight"), qType),
                            quantize(weights.load(mlpPrefix + "down_proj.weight"), qType),
                            quantize(weights.load(mlpPrefix + "up_proj.weight"), qType),
                            configurableTensorProvider
                    );
            LOGGER.debug("loaded qwen3_moe layer={} mlp weights sparse={}", i, sparseLayer);

            blocks[i] = new TransformerBlock(
                    this,
                    i,
                    new RmsNorm(this, quantize(weights.load(base + "input_layernorm.weight"), qType), 0.0f, metricRegistry),
                    attention,
                    new RmsNorm(this, quantize(weights.load(base + "post_attention_layernorm.weight"), qType), 0.0f, metricRegistry),
                    mlp,
                    configurableTensorProvider
            );
            LOGGER.info("loading qwen3_moe layer={} sparse={} done", i, sparseLayer);
        }
        return blocks;
    }

    private io.teknek.deliverance.tensor.AbstractTensor[] expertWeights(int layerIndex, String mlpPrefix, String suffix,
            int numberOfExperts, DType qType) {
        io.teknek.deliverance.tensor.AbstractTensor[] tensors = new io.teknek.deliverance.tensor.AbstractTensor[numberOfExperts];
        for (int expert = 0; expert < numberOfExperts; expert++) {
            if (expert == 0 || expert == numberOfExperts - 1 || expert % 16 == 0) {
                LOGGER.info("loading qwen3_moe layer={} expert_tensor={} expert={}/{}", layerIndex, suffix,
                        expert + 1, numberOfExperts);
            }
            tensors[expert] = quantize(weights.load(mlpPrefix + "experts." + expert + "." + suffix), qType);
        }
        return tensors;
    }
}
