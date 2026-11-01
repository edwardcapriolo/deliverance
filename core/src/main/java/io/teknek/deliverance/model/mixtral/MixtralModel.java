package io.teknek.deliverance.model.mixtral;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.generator2.CausalSelfAttention2;
import io.teknek.deliverance.generator2.FeedForward2;
import io.teknek.deliverance.generator2.MLPBlock2;
import io.teknek.deliverance.generator2.RmsNorm2;
import io.teknek.deliverance.generator2.TransformerBlock2;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.llama.*;
import io.teknek.deliverance.model.MixtureOfExpertsBlock;
import io.teknek.deliverance.model.AbstractModel.InferenceType;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.toolcallparser.ToolCallParser;

import java.util.Optional;
import java.util.stream.IntStream;

import static io.teknek.deliverance.tensor.AbstractTensorUtils.quantize;

public class MixtralModel extends LlamaModel {

    public MixtralModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer t, DType workingMemoryDType,
                        DType workingMemoryQType, Optional<DType> modelQType,
                        MetricRegistry metricRegistry, KvBufferCacheSettings kvBufferCacheSettings,
                        ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
                        TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization,
                        Lighter lighter) {
        super(inferenceType, c, w, t, workingMemoryDType, workingMemoryQType, modelQType, metricRegistry,
                kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext, tensorParallelCollectives,
                outputHeadQuantization, lighter);
    }

    /**
     * Not supported: attention uses plain {@code CausalSelfAttention} (free LoRA coverage), but the
     * feed-forward layer routes through {@code MixtureOfExpertsBlock} (one base tensor name per
     * expert, not per layer) -- the same MoE-routing ambiguity {@code Qwen3MoeModel}/
     * {@code GraniteMoeHybridModel} are excluded for. Overrides {@code LlamaModel}'s inherited
     * {@code true} back to {@code false}. See step 4 plan Section 6/Section 11 item 6 -- this gap
     * was missed by the original family survey and found only during implementation.
     */
    @Override
    protected boolean supportsLoraHotSwap() {
        return false;
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
        if (tensorParallelContext.enabled()) {
            throw new UnsupportedOperationException("TensorRef Mixtral tensor parallelism is not ported");
        }
        MixtralConfig mixtralConfig = (MixtralConfig) config;
        DType qType = modelQType.orElse(this.modelDType);
        TransformerBlock2[] blocks = new TransformerBlock2[config.numberOfLayers];
        IntStream.range(0, config.numberOfLayers).parallel().forEach(i -> {
            String base = "model.layers." + i + ".";
            String attentionPrefix = base + "self_attn.";
            String qName = attentionPrefix + "q_proj.weight";
            String kName = attentionPrefix + "k_proj.weight";
            String vName = attentionPrefix + "v_proj.weight";
            String oName = attentionPrefix + "o_proj.weight";
            CausalSelfAttention2 attention = new CausalSelfAttention2(this, i,
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(qName, weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(kName, weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(vName, weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(oName, weights::loadRef, qType)),
                    lighter, metricRegistry, qName, kName, vName, oName);

            String moePrefix = base + "block_sparse_moe.";
            TensorRef router = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                    moePrefix + "gate.weight", weights::loadRef, qType));
            TensorRef[] gate = loadExpertRefs(moePrefix, "w1.weight", mixtralConfig.numberOfExperts, qType);
            TensorRef[] down = loadExpertRefs(moePrefix, "w2.weight", mixtralConfig.numberOfExperts, qType);
            TensorRef[] up = loadExpertRefs(moePrefix, "w3.weight", mixtralConfig.numberOfExperts, qType);
            FeedForward2 moe = new MixtralMoeFeedForward2(this, mixtralConfig, router, gate, up, down, lighter);

            String inputNormName = base + "input_layernorm.weight";
            String postAttentionNormName = base + "post_attention_layernorm.weight";
            blocks[i] = new TransformerBlock2(this, i,
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(
                            loadAndMaybeQuantizedExcluding1DTensors(inputNormName, weights::loadRef, qType)), 0.0f)),
                    attention, Optional.empty(),
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(
                            loadAndMaybeQuantizedExcluding1DTensors(postAttentionNormName, weights::loadRef, qType)), 0.0f)),
                    moe, Optional.empty(), Optional.empty(), configurableTensorProvider);
        });
        return blocks;
    }

    private TensorRef[] loadExpertRefs(String prefix, String suffix, int experts, DType qType) {
        TensorRef[] result = new TensorRef[experts];
        for (int expert = 0; expert < experts; expert++) {
            result[expert] = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                    prefix + "experts." + expert + "." + suffix, weights::loadRef, qType));
        }
        return result;
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
