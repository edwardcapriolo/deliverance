package io.teknek.deliverance.model.gemma2;



import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.math.FloatConversions;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.TokenRenderer;
import io.teknek.deliverance.model.llama.LlamaModel;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tokenizer.Tokenizer;
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


    public Gemma2Model(
            InferenceType inferenceType,
            Config config,
            WeightLoader weights,
            Tokenizer tokenizer,
            DType workingDType,
            DType workingQType,
            Optional<DType> modelQType, ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry,
            TensorAllocator arrayQueueTensorAllocator, KvBufferCacheSettings kvBufferCacheSettings, TokenRenderer tokenRenderer,
            ToolCallParser toolCallParser, WrappedForkJoinPool pool
    ) {
        super(inferenceType, config, weights, tokenizer, workingDType, workingQType, modelQType,
                configurableTensorProvider, metricRegistry, arrayQueueTensorAllocator, kvBufferCacheSettings, tokenRenderer, toolCallParser, pool);
        // https://github.com/huggingface/transformers/blob/1082361a1978d30db5c3932d1ee08914d74d9697/src/transformers/models/gemma/modeling_gemma.py#L898
        // This is the scaling factor for the embedding layer but google's implementation is a is rounded to 16 bits
        this.embeddingScalingFactor = FloatConversions.bFloat16ToFloat32(
                FloatConversions.float32ToBFloat16((float) Math.pow(config.embeddingLength, 0.5))
        );
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        TransformerBlock[] transformerBlocks = new TransformerBlock[config.dctx().numberOfLayers];
        IntStream.range(config.dctx().layerStart, config.dctx().layerEnd).parallel().forEach(i -> {
            String base = "model.layers." + i + ".";
            String prefix = base + "self_attn.";
            CausalSelfAttention attention = new CausalSelfAttention(this, i,
                    quantize(weights.load(prefix + "q_proj.weight", config.dctx(), true, false), qType),
                    quantize(weights.load(prefix + "k_proj.weight", config.dctx(), true, false), qType),
                    quantize(weights.load(prefix + "v_proj.weight", config.dctx(), true, false), qType),
                    quantize(weights.load(prefix + "o_proj.weight", config.dctx(), false, true), qType),
                    configurableTensorProvider,
                    metricRegistry
            );

            prefix = base + "mlp.";
            MLPBlock mlp = new MLPBlock(
                    this,
                    config.activationFunction,
                    quantize(weights.load(prefix + "gate_proj.weight", config.dctx(), true, false), qType), // w1
                    quantize(weights.load(prefix + "down_proj.weight", config.dctx(), false, true), qType), // w2
                    quantize(weights.load(prefix + "up_proj.weight", config.dctx(), true, false), qType), // w3,
                    configurableTensorProvider
            );

            transformerBlocks[i] = new TransformerBlock(this, i,
                    new RmsNorm(this, quantize(weights.load(base + "input_layernorm.weight"), qType), 1.0f, metricRegistry),
                    attention,
                    new RmsNorm(this, quantize(weights.load(base + "post_attention_layernorm.weight"), qType), 1.0f, metricRegistry),
                    new RmsNorm(this, quantize(weights.load(base + "pre_feedforward_layernorm.weight"), qType), 1.0f, metricRegistry),
                    mlp,
                    new RmsNorm(this, quantize(weights.load(base + "post_feedforward_layernorm.weight"), qType), 1.0f, metricRegistry),
                    configurableTensorProvider
            );

        });

        return transformerBlocks;
    }

    @Override
    protected EmbedInput loadInputWeights() {
        // this comment is it true or not?
        // Don't quantize this, it's used for the embedding layer
        if (wte == null) {
            wte = quantize(weights.load("model.embed_tokens.weight"), workingDType);
        }
        return new EmbedInput(this) {
            @Override
            public AbstractTensor inputTokenToEmbedding(int inputToken, int position) {
                    AbstractTensor embedding = makeDenseTensor(config.embeddingLength);
                    AbstractTensor at = wte.slice(true, inputToken);
                    if (wte.dType() != embedding.dType()) {
                        at = configurableTensorProvider.get().quantize(at, embedding.dType(), 0, config.embeddingLength);
                    }
                    embedding.copyFrom(at, 0, 0, config.embeddingLength);
                    // This is important for Gemma, but not for Llama
                    configurableTensorProvider.get().scale(embeddingScalingFactor, embedding, 0, config.embeddingLength);
                    return embedding;
            }
        };
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        if (wte == null) {
            wte = quantize( weights.load("model.embed_tokens.weight"), workingDType);
        }
        final LayerNorm layerNorm = new RmsNorm(this, quantize( weights.load("model.norm.weight"), qType),
                1.0f, metricRegistry);
        return new SampleOutput() {
            @Override
            public LayerNorm getOutputLayerNorm() {
                return layerNorm;
            }

            @Override
            public AbstractTensor getOutputLogitsWeights() {
                return wte;
            }
        };
    }
}