package io.teknek.deliverance.model.gemma2;



import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.generator2.CausalSelfAttention2;
import io.teknek.deliverance.generator2.MLPBlock2;
import io.teknek.deliverance.generator2.RmsNorm2;
import io.teknek.deliverance.generator2.TransformerBlock2;
import io.teknek.deliverance.generator2.LayerNorm2;
import io.teknek.deliverance.generator2.SampleOutputRef;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.FloatConversions;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.llama.LlamaModel;
import io.teknek.deliverance.model.tensorparallel.DefaultTransformerWeightPolicyResolver;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelPlanner;
import io.teknek.deliverance.model.tensorparallel.TensorParallelShardPlan;
import io.teknek.deliverance.model.tensorparallel.TensorParallelWeightLoader;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor2.TensorRefBackedTensor;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.stream.IntStream;

import static io.teknek.deliverance.tensor.AbstractTensorUtils.quantize;

public class Gemma2Model extends LlamaModel {
    private static final Logger logger = LoggerFactory.getLogger(Gemma2Model.class);

    private final float embeddingScalingFactor;
    private AbstractTensor wte;
    private AbstractTensor outputLogitsWeights;
    private TensorRef wteRef;

    public Gemma2Model(
            InferenceType inferenceType,
            Config config,
            WeightLoader weights,
            PreTrainedTokenizer tokenizer,
            DType workingDType,
            DType workingQType,
            Optional<DType> modelQType, MetricRegistry metricRegistry, KvBufferCacheSettings kvBufferCacheSettings,
            ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives,
            Optional<DType> outputHeadQuantization, Lighter lighter
    ) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType,
                metricRegistry, kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext,
                tensorParallelCollectives, outputHeadQuantization, lighter);
        // https://github.com/huggingface/transformers/blob/1082361a1978d30db5c3932d1ee08914d74d9697/src/transformers/models/gemma/modeling_gemma.py#L898
        // This is the scaling factor for the embedding layer but google's implementation is a is rounded to 16 bits
        this.embeddingScalingFactor = FloatConversions.bFloat16ToFloat32(
                FloatConversions.float32ToBFloat16((float) Math.pow(config.embeddingLength, 0.5))
        );
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
            CausalSelfAttention2 attention = new CausalSelfAttention2(this, i,
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(qName,
                            tensorParallelWeights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(kName,
                            tensorParallelWeights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(vName,
                            tensorParallelWeights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(oName,
                            tensorParallelWeights::loadRef, qType)),
                    lighter, metricRegistry, qName, kName, vName, oName);

            String mlp = base + "mlp.";
            String gateName = mlp + "gate_proj.weight";
            String downName = mlp + "down_proj.weight";
            String upName = mlp + "up_proj.weight";
            MLPBlock2 feedForward = new MLPBlock2(this,
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(gateName,
                            tensorParallelWeights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(upName,
                            tensorParallelWeights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(downName,
                            tensorParallelWeights::loadRef, qType)),
                    lighter, gateName, upName, downName);

            blocks[i] = new TransformerBlock2(this, i,
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(
                            loadAndMaybeQuantizedExcluding1DTensors(base + "input_layernorm.weight",
                                    weights::loadRef, qType)), 1.0f)),
                    attention,
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(
                            loadAndMaybeQuantizedExcluding1DTensors(base + "post_attention_layernorm.weight",
                                    weights::loadRef, qType)), 1.0f)),
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(
                            loadAndMaybeQuantizedExcluding1DTensors(base + "pre_feedforward_layernorm.weight",
                                    weights::loadRef, qType)), 1.0f)),
                    feedForward,
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(
                            loadAndMaybeQuantizedExcluding1DTensors(base + "post_feedforward_layernorm.weight",
                                    weights::loadRef, qType)), 1.0f)),
                     Optional.empty());
        });
        return blocks;
    }

    @Override
    protected SampleOutputRef loadOutputWeightsRef() {
        DType qType = modelQType.orElse(this.modelDType);
        TensorRef norm = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors("model.norm.weight",
                weights::loadRef, qType));
        TensorRef head = wteRef;
        return new SampleOutputRef() {
            @Override
            public LayerNorm2 outputLayerNorm() {
                return new RmsNorm2(Gemma2Model.this, norm, 1.0f);
            }

            @Override
            public TensorRef outputLogitsWeights() {
                return head;
            }
        };
    }

    @Override
    protected EmbedInput loadInputWeights() {
        if (usesTensorRefExecution()) {
            wteRef = registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(
                    "model.embed_tokens.weight", weights::loadRef, workingDType));
            return new EmbedInput(this) {
                @Override
                public TensorRef inputTokenToEmbedding(int inputToken, int position) {
                    try (TensorRef row = wteRef.slice(inputToken)) {
                        TensorRef embedding = makeDenseTensorRef(1, config.embeddingLength);
                        try {
                            if (row.dType() == embedding.dType()) {
                                lighter.copy(row, 0, embedding, 0, config.embeddingLength);
                            } else {
                                try (TensorRef converted = lighter.reshape(row, embedding.dType())) {
                                    lighter.copy(converted, 0, embedding, 0, config.embeddingLength);
                                }
                            }
                            lighter.scale(new io.teknek.deliverance.tensor2.Scale(embeddingScalingFactor)
                                    .target(embedding).offsetAndLength(0, config.embeddingLength));
                            return embedding;
                        } catch (RuntimeException | Error e) {
                            embedding.close();
                            throw e;
                        }
                    }
                }
            };
        }
        // this comment is it true or not?
        // Don't quantize this, it's used for the embedding layer
        if (wte == null) {
            wte = quantize(weights.load("model.embed_tokens.weight"), workingDType);
        }
        return new EmbedInput(this) {
            @Override
            public TensorRef inputTokenToEmbedding(int inputToken, int position) {
                    AbstractTensor embedding = new TensorRefBackedTensor(makeDenseTensorRef(config.embeddingLength));
                    AbstractTensor at = wte.slice(true, inputToken);
                    if (wte.dType() != embedding.dType()) {
                        at = quantize(at, embedding.dType());
                    }
                    embedding.copyFrom(at, 0, 0, config.embeddingLength);
                    // This is important for Gemma, but not for Llama
                    scale(embeddingScalingFactor, embedding, 0, config.embeddingLength);
                    return TensorRef.owned(embedding);
            }
        };
    }

    private AbstractTensor loadOutputLogitsWeights(DType outputHeadDType, boolean force) {
        AbstractTensor original = weights.load("model.embed_tokens.weight");
        AbstractTensor result = force
                ? io.teknek.deliverance.tensor.AbstractTensorUtils.quantize(original, outputHeadDType, true)
                : quantize(original, outputHeadDType);
        if (result != original) {
            original.close();
        }
        return result;
    }
}
