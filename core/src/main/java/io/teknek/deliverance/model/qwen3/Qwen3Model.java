package io.teknek.deliverance.model.qwen3;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.MLPBlock;
import io.teknek.deliverance.generator.Qwen3CausalSelfAttention;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.generator.RmsNorm;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.llama.LlamaModel;
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

public class Qwen3Model extends LlamaModel {

    public Qwen3Model(
            InferenceType inferenceType,
            Config config,
            WeightLoader weights,
            PreTrainedTokenizer tokenizer,
            DType workingDType,
            DType workingQType,
            Optional<DType> modelQType,
            ConfigurableTensorProvider configurableTensorProvider,
            MetricRegistry metricRegistry,
            TensorAllocator arrayQueueTensorAllocator,
            KvBufferCacheSettings kvBufferCacheSettings,
            ToolCallParser toolCallParser,
            WrappedForkJoinPool pool,
            TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives,
            Optional<DType> outputHeadQuantization
    ) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType,
                configurableTensorProvider, metricRegistry, arrayQueueTensorAllocator, kvBufferCacheSettings,
                toolCallParser, pool, tensorParallelContext, tensorParallelCollectives, outputHeadQuantization);
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        TransformerBlock[] blocks = new TransformerBlock[config.numberOfLayers];
        IntStream.range(0, config.numberOfLayers).parallel().forEach(i -> {
            String base = "model.layers." + i + ".";
            String attn = base + "self_attn.";
            String qName = attn + "q_proj.weight";
            String kName = attn + "k_proj.weight";
            String vName = attn + "v_proj.weight";
            String oName = attn + "o_proj.weight";
            Qwen3CausalSelfAttention attention = new Qwen3CausalSelfAttention(
                    this,
                    i,
                    quantize(weights.load(qName), qType),
                    quantize(weights.load(kName), qType),
                    quantize(weights.load(vName), qType),
                    quantize(weights.load(oName), qType),
                    quantize(weights.load(attn + "q_norm.weight"), qType),
                    quantize(weights.load(attn + "k_norm.weight"), qType),
                    configurableTensorProvider,
                    metricRegistry,
                    qName, kName, vName, oName
            );

            String mlpPrefix = base + "mlp.";
            String gateName = mlpPrefix + "gate_proj.weight";
            String downName = mlpPrefix + "down_proj.weight";
            String upName = mlpPrefix + "up_proj.weight";
            MLPBlock mlp = new MLPBlock(
                    this,
                    config.activationFunction,
                    quantize(weights.load(gateName), qType),
                    quantize(weights.load(downName), qType),
                    quantize(weights.load(upName), qType),
                    configurableTensorProvider,
                    gateName, upName, downName
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

    @Override
    protected Response postProcessResponse(Response response) {
        QwenReasoningParser.Parsed parsed = QwenReasoningParser.parse(response.responseTextWithSpecialTokens,
                response.responseText);
        return response.copyWithText(parsed.content(), response.responseTextWithSpecialTokens, parsed.reasoning());
    }
}
