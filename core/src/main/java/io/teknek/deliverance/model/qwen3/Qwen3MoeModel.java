package io.teknek.deliverance.model.qwen3;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.MLPBlock;
import io.teknek.deliverance.generator.Qwen3CausalSelfAttention;
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

import java.util.Optional;
import java.util.stream.IntStream;

import static io.teknek.deliverance.tensor.AbstractTensorUtils.quantize;

public class Qwen3MoeModel extends Qwen3Model {
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

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        Qwen3MoeConfig moeConfig = (Qwen3MoeConfig) config;
        DType qType = modelQType.orElse(this.modelDType);
        TransformerBlock[] blocks = new TransformerBlock[config.numberOfLayers];
        IntStream.range(0, config.numberOfLayers).parallel().forEach(i -> {
            String base = "model.layers." + i + ".";
            String attn = base + "self_attn.";
            Qwen3CausalSelfAttention attention = new Qwen3CausalSelfAttention(
                    this,
                    i,
                    quantize(weights.load(attn + "q_proj.weight"), qType),
                    quantize(weights.load(attn + "k_proj.weight"), qType),
                    quantize(weights.load(attn + "v_proj.weight"), qType),
                    quantize(weights.load(attn + "o_proj.weight"), qType),
                    quantize(weights.load(attn + "q_norm.weight"), qType),
                    quantize(weights.load(attn + "k_norm.weight"), qType),
                    configurableTensorProvider,
                    metricRegistry
            );

            String mlpPrefix = base + "mlp.";
            var mlp = moeConfig.sparseLayer(i)
                    ? new Qwen3MoeFeedForward(this, moeConfig,
                            quantize(weights.load(mlpPrefix + "gate.weight"), qType),
                            quantize(weights.load(mlpPrefix + "experts.gate_up_proj"), qType),
                            quantize(weights.load(mlpPrefix + "experts.down_proj"), qType),
                            configurableTensorProvider)
                    : new MLPBlock(
                            this,
                            config.activationFunction,
                            quantize(weights.load(mlpPrefix + "gate_proj.weight"), qType),
                            quantize(weights.load(mlpPrefix + "down_proj.weight"), qType),
                            quantize(weights.load(mlpPrefix + "up_proj.weight"), qType),
                            configurableTensorProvider
                    );

            blocks[i] = new TransformerBlock(
                    this,
                    i,
                    new RmsNorm(this, quantize(weights.load(base + "input_layernorm.weight"), qType), 0.0f, metricRegistry),
                    attention,
                    new RmsNorm(this, quantize(weights.load(base + "post_attention_layernorm.weight"), qType), 0.0f, metricRegistry),
                    mlp,
                    configurableTensorProvider
            );
        });
        return blocks;
    }
}
