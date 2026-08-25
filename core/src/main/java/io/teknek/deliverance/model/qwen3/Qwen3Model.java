package io.teknek.deliverance.model.qwen3;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.MLPBlock;
import io.teknek.deliverance.generator.Qwen3CausalSelfAttention;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.generator.RmsNorm;
import io.teknek.deliverance.generator.TransformerBlock;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.llama.LlamaModel;
import io.teknek.deliverance.model.tensorparallel.DefaultTransformerWeightPolicyResolver;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.TensorParallelPlanner;
import io.teknek.deliverance.model.tensorparallel.TensorParallelShardPlan;
import io.teknek.deliverance.model.tensorparallel.TensorParallelWeightLoader;
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
        TensorParallelShardPlan tensorParallelPlan = TensorParallelPlanner.plan(config, tensorParallelContext);
        TensorParallelWeightLoader tensorParallelWeights = new TensorParallelWeightLoader(weights,
                tensorParallelContext, tensorParallelPlan, new DefaultTransformerWeightPolicyResolver());
        TransformerBlock[] blocks = new TransformerBlock[config.numberOfLayers];
        IntStream.range(0, config.numberOfLayers).parallel().forEach(i -> {
            String base = "model.layers." + i + ".";
            String attn = base + "self_attn.";
            String qName = attn + "q_proj.weight";
            String kName = attn + "k_proj.weight";
            String vName = attn + "v_proj.weight";
            String oName = attn + "o_proj.weight";
            String qNormName = attn + "q_norm.weight";
            String kNormName = attn + "k_norm.weight";
            var qWeight = quantize(tensorParallelWeights.load(qName), qType);
            var kWeight = quantize(tensorParallelWeights.load(kName), qType);
            var vWeight = quantize(tensorParallelWeights.load(vName), qType);
            var oWeight = quantize(tensorParallelWeights.load(oName), qType);
            var qNormWeight = quantize(weights.load(qNormName), qType);
            var kNormWeight = quantize(weights.load(kNormName), qType);
            registerModelLineageTensor(qName, qWeight);
            registerModelLineageTensor(kName, kWeight);
            registerModelLineageTensor(vName, vWeight);
            registerModelLineageTensor(oName, oWeight);
            registerModelLineageTensor(qNormName, qNormWeight);
            registerModelLineageTensor(kNormName, kNormWeight);
            Qwen3CausalSelfAttention attention = new Qwen3CausalSelfAttention(
                    this,
                    i,
                    qWeight,
                    kWeight,
                    vWeight,
                    oWeight,
                    qNormWeight,
                    kNormWeight,
                    configurableTensorProvider,
                    metricRegistry,
                    qName, kName, vName, oName
            );

            String mlpPrefix = base + "mlp.";
            String gateName = mlpPrefix + "gate_proj.weight";
            String downName = mlpPrefix + "down_proj.weight";
            String upName = mlpPrefix + "up_proj.weight";
            var gateWeight = quantize(tensorParallelWeights.load(gateName), qType);
            var downWeight = quantize(tensorParallelWeights.load(downName), qType);
            var upWeight = quantize(tensorParallelWeights.load(upName), qType);
            registerModelLineageTensor(gateName, gateWeight);
            registerModelLineageTensor(downName, downWeight);
            registerModelLineageTensor(upName, upWeight);
            MLPBlock mlp = new MLPBlock(
                    this,
                    config.activationFunction,
                    gateWeight,
                    downWeight,
                    upWeight,
                    configurableTensorProvider,
                    "layer." + i + ".mlp.down_proj",
                    gateName, upName, downName,
                    i
            );

            String inputNormName = base + "input_layernorm.weight";
            String postAttentionNormName = base + "post_attention_layernorm.weight";
            var inputNormWeight = quantize(weights.load(inputNormName), qType);
            var postAttentionNormWeight = quantize(weights.load(postAttentionNormName), qType);
            registerModelLineageTensor(inputNormName, inputNormWeight);
            registerModelLineageTensor(postAttentionNormName, postAttentionNormWeight);

            blocks[i] = new TransformerBlock(
                    this,
                    i,
                    new RmsNorm(this, inputNormWeight, 0.0f, metricRegistry),
                    attention,
                    new RmsNorm(this, postAttentionNormWeight, 0.0f, metricRegistry),
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
