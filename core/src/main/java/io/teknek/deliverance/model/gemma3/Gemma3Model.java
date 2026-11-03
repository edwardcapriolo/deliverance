package io.teknek.deliverance.model.gemma3;



import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.generator2.CausalSelfAttention2;
import io.teknek.deliverance.generator2.FeedForward2;
import io.teknek.deliverance.generator2.LayerNorm2;
import io.teknek.deliverance.generator2.MLPBlock2;
import io.teknek.deliverance.generator2.RmsNorm2;
import io.teknek.deliverance.generator2.SampleOutputRef;
import io.teknek.deliverance.generator2.TransformerBlock2;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.FloatConversions;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.llama.LlamaModel;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.Scale;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.stream.IntStream;

import static io.teknek.deliverance.tensor.AbstractTensorUtils.quantize;

public class Gemma3Model extends LlamaModel {
    private static final Logger logger = LoggerFactory.getLogger(Gemma3Model.class);

    private final float embeddingScalingFactor;
    private AbstractTensor wte;


    public Gemma3Model(InferenceType inferenceType, Config config, WeightLoader weights,
            PreTrainedTokenizer tokenizer, DType workingDType, DType workingQType, Optional<DType> modelQType,
            MetricRegistry metricRegistry, KvBufferCacheSettings kvBufferCacheSettings, ToolCallParser toolCallParser,
            WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization,
            Lighter lighter) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType, metricRegistry,
                kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext, tensorParallelCollectives,
                outputHeadQuantization, lighter);
        this.embeddingScalingFactor = FloatConversions.bFloat16ToFloat32(FloatConversions.float32ToBFloat16(
                (float) Math.pow(config.embeddingLength, 0.5)));
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
    protected EmbedInput loadInputWeights() {
        TensorRef embeddingWeights = registerModelTensorRef(weights.loadRef("model.embed_tokens.weight"));
        return new EmbedInput(this) {
            @Override
            public TensorRef inputTokenToEmbedding(int inputToken, int position) {
                try (TensorRef row = embeddingWeights.slice(inputToken)) {
                    TensorRef embedding = makeDenseTensorRef(1, config.embeddingLength);
                    try {
                        if (row.dType() != workingDType) {
                            try (TensorRef converted = lighter.reshape(row, workingDType)) {
                                lighter.copy(converted, 0, embedding, 0, config.embeddingLength);
                            }
                        } else {
                            lighter.copy(row, 0, embedding, 0, config.embeddingLength);
                        }
                        lighter.scale(new Scale(embeddingScalingFactor).target(embedding)
                                .offsetAndLength(0, config.embeddingLength));
                        return embedding;
                    } catch (RuntimeException | Error e) {
                        embedding.close();
                        throw e;
                    }
                }
            }
        };
    }

    @Override
    protected TransformerBlock2[] loadTransformerBlockWeights2() {
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
            String mlpPrefix = base + "mlp.";
            String gateName = mlpPrefix + "gate_proj.weight";
            String downName = mlpPrefix + "down_proj.weight";
            String upName = mlpPrefix + "up_proj.weight";
            FeedForward2 mlp = new MLPBlock2(this,
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(gateName, weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(upName, weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(downName, weights::loadRef, qType)),
                    lighter, gateName, upName, downName);
            String inputNorm = base + "input_layernorm.weight";
            String postAttentionNorm = base + "post_attention_layernorm.weight";
            String preFeedForwardNorm = base + "pre_feedforward_layernorm.weight";
            String postFeedForwardNorm = base + "post_feedforward_layernorm.weight";
            blocks[i] = new TransformerBlock2(this, i,
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(
                            loadAndMaybeQuantizedExcluding1DTensors(inputNorm, weights::loadRef, qType)), 1.0f)),
                    attention,
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                            postAttentionNorm, weights::loadRef, qType)), 1.0f)),
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                            preFeedForwardNorm, weights::loadRef, qType)), 1.0f)),
                    mlp,
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                            postFeedForwardNorm, weights::loadRef, qType)), 1.0f)),
                     Optional.empty());
        });
        return blocks;
    }


    @Override
    protected SampleOutputRef loadOutputWeightsRef() {
        DType qType = modelQType.orElse(this.modelDType);
        TensorRef norm = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                "model.norm.weight", weights::loadRef, qType));
        TensorRef head = weights.loadRef("model.embed_tokens.weight");
        DType outputDType = outputHeadQuantization.orElse(workingDType);
        if (head.dType() != outputDType) {
            TensorRef converted = lighter.reshape(head, outputDType);
            head.close();
            head = converted;
        }
        TensorRef outputHead = registerModelTensorRef(head);
        return new SampleOutputRef() {
            @Override
            public LayerNorm2 outputLayerNorm() {
                return new RmsNorm2(Gemma3Model.this, norm, 1.0f);
            }

            @Override
            public TensorRef outputLogitsWeights() {
                return outputHead;
            }
        };
    }

}
