package io.teknek.deliverance.model.qwen3;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator2.MLPBlock2;
import io.teknek.deliverance.generator2.Qwen3KvCacheSelfAttention2;
import io.teknek.deliverance.generator2.RmsNorm2;
import io.teknek.deliverance.generator2.TransformerBlock2;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.DefaultTransformerWeightPolicyResolver;
import io.teknek.deliverance.model.tensorparallel.TensorParallelPlanner;
import io.teknek.deliverance.model.tensorparallel.TensorParallelShardPlan;
import io.teknek.deliverance.model.tensorparallel.TensorParallelWeightLoader;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.stream.IntStream;

public class Qwen3MoeModel extends Qwen3Model {
    private static final Logger LOGGER = LoggerFactory.getLogger(Qwen3MoeModel.class);

    public Qwen3MoeModel(InferenceType inferenceType, Config config, WeightLoader weights,
            PreTrainedTokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType,
            MetricRegistry metricRegistry, KvBufferCacheSettings kvBufferCacheSettings,
            ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization,
            Lighter lighter) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType,
                metricRegistry, kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext,
                tensorParallelCollectives, outputHeadQuantization, lighter);
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
    protected TransformerBlock2[] loadTransformerBlockWeights2() {
        Qwen3MoeConfig moeConfig = (Qwen3MoeConfig) config;
        DType qType = modelQType.orElse(this.modelDType);
        TensorParallelShardPlan tensorParallelPlan = TensorParallelPlanner.plan(config, tensorParallelContext);
        TensorParallelWeightLoader tensorParallelWeights = new TensorParallelWeightLoader(weights,
                tensorParallelContext, tensorParallelPlan, new DefaultTransformerWeightPolicyResolver());
        TransformerBlock2[] blocks = new TransformerBlock2[config.numberOfLayers];
        IntStream.range(0, config.numberOfLayers).parallel().forEach(i -> {
            String base = "model.layers." + i + ".";
            String attn = base + "self_attn.";
            String qName = attn + "q_proj.weight";
            String kName = attn + "k_proj.weight";
            String vName = attn + "v_proj.weight";
            String oName = attn + "o_proj.weight";
            TensorRef qRef = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(qName,
                    tensorParallelWeights::loadRef, qType));
            TensorRef kRef = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(kName,
                    tensorParallelWeights::loadRef, qType));
            TensorRef vRef = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(vName,
                    tensorParallelWeights::loadRef, qType));
            TensorRef oRef = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(oName,
                    tensorParallelWeights::loadRef, qType));
            String qNormName = attn + "q_norm.weight";
            String kNormName = attn + "k_norm.weight";
            TensorRef qNormRef = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(qNormName,
                    weights::loadRef, qType));
            TensorRef kNormRef = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(kNormName,
                    weights::loadRef, qType));
            Qwen3KvCacheSelfAttention2 attention = new Qwen3KvCacheSelfAttention2(this, i, qRef, kRef, vRef, oRef,
                    qNormRef, kNormRef, lighter, metricRegistry, qName, kName, vName, oName);

            String mlpPrefix = base + "mlp.";
            io.teknek.deliverance.generator2.FeedForward2 mlp;
            if (moeConfig.sparseLayer(i)) {
                TensorRef router = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                        mlpPrefix + "gate.weight", weights::loadRef, qType));
                TensorRef[] gate = loadExpertRefs(i, mlpPrefix, "gate_proj.weight", moeConfig.numExperts, qType);
                TensorRef[] up = loadExpertRefs(i, mlpPrefix, "up_proj.weight", moeConfig.numExperts, qType);
                TensorRef[] down = loadExpertRefs(i, mlpPrefix, "down_proj.weight", moeConfig.numExperts, qType);
                mlp = new Qwen3MoeFeedForward2(this, moeConfig, router, gate, up, down, lighter);
            } else {
                String gateName = mlpPrefix + "gate_proj.weight";
                String downName = mlpPrefix + "down_proj.weight";
                String upName = mlpPrefix + "up_proj.weight";
                mlp = new MLPBlock2(this,
                        registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(gateName,
                                tensorParallelWeights::loadRef, qType)),
                        registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(upName,
                                tensorParallelWeights::loadRef, qType)),
                        registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(downName,
                                tensorParallelWeights::loadRef, qType)), lighter, gateName, upName, downName);
            }

            String inputNormName = base + "input_layernorm.weight";
            String postAttentionNormName = base + "post_attention_layernorm.weight";
            TensorRef inputNorm = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(inputNormName,
                    weights::loadRef, qType));
            TensorRef postAttentionNorm = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                    postAttentionNormName, weights::loadRef, qType));
            blocks[i] = new TransformerBlock2(this, i, new RmsNorm2(this, inputNorm, 0.0f), attention,
                    new RmsNorm2(this, postAttentionNorm, 0.0f), mlp);
        });
        return blocks;
    }

    private TensorRef[] loadExpertRefs(int layerIndex, String mlpPrefix, String suffix, int experts, DType qType) {
        TensorRef[] result = new TensorRef[experts];
        for (int expert = 0; expert < experts; expert++) {
            result[expert] = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                    mlpPrefix + "experts." + expert + "." + suffix, weights::loadRef, qType));
        }
        return result;
    }
}
