package io.teknek.deliverance.model.llama;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.TensorOperationsProvider;
import io.teknek.deliverance.tokenizer.Tokenizer;

import java.util.Optional;
import java.util.stream.IntStream;

public class LlamaModel extends AbstractModel {

    private volatile AbstractTensor embedTokenWeights;
    public LlamaModel(InferenceType inferenceType, Config c, WeightLoader w, Tokenizer t, DType workingMemoryDType,
                      DType workingMemoryQType, Optional<DType> modelQType, ConfigurableTensorProvider configurableTensorProvider) {
        super(inferenceType, c, w, t, workingMemoryDType, workingMemoryQType, modelQType, configurableTensorProvider);
    }

    @Override
    protected EmbedInput loadInputWeights() {

        // Don't quantize this, it's used for the embedding layer
        // but we ae calling quantize in the if?
        if (embedTokenWeights == null) {
            embedTokenWeights = weights.load("model.embed_tokens.weight").quantize(workingDType);
            configurableTensorProvider.get().registerModelTensor(embedTokenWeights);
        }

        return (inputToken, position) -> {
            if (embedTokenWeights.dType() == DType.BF16) {
                // Handle old style model with BF16 embeddings
                AbstractTensor embedding = makeDenseTensor(1, config.embeddingLength);
                AbstractTensor at = embedTokenWeights.slice(true, inputToken);
                if (embedTokenWeights.dType() != embedding.dType()) {
                    at = TensorOperationsProvider.get().quantize(at, embedding.dType(), 0, config.embeddingLength);
                }
                embedding.copyFrom(at, 0, 0, config.embeddingLength);
                return embedding;
            } else {
                AbstractTensor at = embedTokenWeights.slice(true, inputToken);
                AbstractTensor embedding = at.copyShape();
                embedding.copyFrom(at, 0, 0, config.embeddingLength);
                return embedding;
            }
        };
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        if (qType != this.modelDType) {
            //logger.info("Quantizing model with {} - Please hold...", qType);
        }
        TransformerBlock[] transformerBlocks = new TransformerBlock[config.dctx().numberOfLayers];
        IntStream.range(config.dctx().layerStart, config.dctx().layerEnd).parallel().forEach(i -> {
            int relativeLayer = i - config.dctx().layerStart; // FIXME: add a helper to the context
            String base = "model.layers." + i + ".";
            String prefix = base + "self_attn.";
            CausalSelfAttention attention = new CausalSelfAttention(
                    this,
                    relativeLayer,
                    weights.load(prefix + "q_proj.weight", config.dctx(), true, false)/*.quantize(qType)*/,
                    weights.load(prefix + "k_proj.weight", config.dctx(), true, false)/*.quantize(qType)*/,
                    weights.load(prefix + "v_proj.weight", config.dctx(), true, false)/*.quantize(qType)*/,
                    weights.load(prefix + "o_proj.weight", config.dctx(), false, true)/*.quantize(qType)*/
            );

            prefix = base + "mlp.";

            MLPBlock mlp = new MLPBlock(
                    this,
                    config.activationFunction,
                    weights.load(prefix + "gate_proj.weight", config.dctx(), true, false)/*.quantize(qType)*/, // w1
                    weights.load(prefix + "down_proj.weight", config.dctx(), false, true)/*.quantize(qType)*/, // w2
                    weights.load(prefix + "up_proj.weight", config.dctx(), true, false)/*.quantize(qType)*/
            ); // w3

            transformerBlocks[relativeLayer] = new TransformerBlock(
                    this,
                    relativeLayer,
                    new RmsNorm(this, weights.load(base + "input_layernorm.weight")/*.quantize(qType)*/),
                    attention,
                    new RmsNorm(this, weights.load(base + "post_attention_layernorm.weight")/*.quantize(qType)*/),
                    mlp
            );
        });

        return transformerBlocks;
    }




    @Override
    protected SampleOutput loadOutputWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        final LayerNorm outputLayerNorm = new RmsNorm(this, weights.load("model.norm.weight").quantize(qType));
        // Some llama models don't have a classification head
        AbstractTensor classificationWeights = weights.isWeightPresent("lm_head.weight")
                ? weights.load("lm_head.weight").quantize(workingDType)
                : embedTokenWeights == null ? embedTokenWeights = weights.load("model.embed_tokens.weight")
                : embedTokenWeights;
        configurableTensorProvider.get().registerModelTensor(classificationWeights);
        return new SampleOutput() {
            @Override
            public LayerNorm getOutputLayerNorm() {
                return outputLayerNorm;
            }

            @Override
            public AbstractTensor getOutputLogitsWeights() {
                return classificationWeights;
            }
        };
    }

    public AbstractTensor maybeQuantize(AbstractTensor t) {
        Preconditions.checkArgument(t.dims() == 2, "Unexpected shape");
        if (t.dType() == workingQType) {
            return super.maybeQuantize(t);
        }
        return configurableTensorProvider.get().quantize(t, workingQType, 0, Ints.checkedCast(t.shape().last()));
    }
}