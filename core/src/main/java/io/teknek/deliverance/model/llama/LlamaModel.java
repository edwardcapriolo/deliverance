package io.teknek.deliverance.model.llama;

import com.codahale.metrics.MetricRegistry;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.stream.IntStream;

import static io.teknek.deliverance.tensor.AbstractTensorUtils.quantize;

public class LlamaModel extends AbstractModel {
    private static final Logger LOGGER = LoggerFactory.getLogger(LlamaModel.class);

    private volatile AbstractTensor embedTokenWeights;
    public LlamaModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer t, DType workingMemoryDType,
                       DType workingMemoryQType, Optional<DType> modelQType,
                      ConfigurableTensorProvider configurableTensorProvider, MetricRegistry metricRegistry,
                      TensorAllocator arrayQueueTensorAllocator, KvBufferCacheSettings kvBufferCacheSettings,
                      ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
                      TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization) {
        super(inferenceType, c, w, t, workingMemoryDType, workingMemoryQType, modelQType, configurableTensorProvider,
                metricRegistry, arrayQueueTensorAllocator, kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext,
                tensorParallelCollectives, outputHeadQuantization);
    }

    @Override
    protected EmbedInput loadInputWeights() {

        //TODO resolvethis
        // Don't quantize this, it's used for the embedding layer
        // but we ae calling quantize in the if?
        if (embedTokenWeights == null) {
            //embedTokenWeights = weights.load("model.embed_tokens.weight").quantize(workingDType);
            LOGGER.debug("loading input embeddings weight=model.embed_tokens.weight target_dtype={}", workingDType);
            embedTokenWeights = quantize(weights.load("model.embed_tokens.weight"), workingDType);
            LOGGER.debug("loaded input embeddings shape={} dtype={}", embedTokenWeights.shape(), embedTokenWeights.dType());
            registerModelLineageTensor("model.embed_tokens.weight", embedTokenWeights);
            configurableTensorProvider.get().registerModelTensor(embedTokenWeights);
        }

        return new EmbedInput(this) {
            @Override
            //TODO The second argument position was  double check that this is propper
            public AbstractTensor inputTokenToEmbedding(int inputToken, int unused) {
                if (embedTokenWeights.dType() == DType.BF16) {
                    // Handle old style model with BF16 embeddings
                    AbstractTensor embedding = makeDenseTensor(1, config.embeddingLength);
                    AbstractTensor at = embedTokenWeights.slice(true, inputToken);
                    if (embedTokenWeights.dType() != embedding.dType()) {
                        at = configurableTensorProvider.get().quantize(at, embedding.dType(), 0, config.embeddingLength);
                    }
                    embedding.copyFrom(at, 0, 0, config.embeddingLength);
                    return embedding;
                } else {
                    AbstractTensor at = embedTokenWeights.slice(true, inputToken);
                    AbstractTensor embedding = parent.getTensorAllocator().getDirty(at.dType(), at.shape());
                    embedding.copyFrom(at, 0, 0, config.embeddingLength);
                    return embedding;
                }
            }
        };
    }

    /**
     * Supports LoRA runtime hot-swap: {@code loadTransformerBlockWeights()} threads real per-layer
     * base tensor names through plain {@code CausalSelfAttention}/{@code MLPBlock} construction.
     * Inherited by every {@code LlamaModel} subclass unless overridden back to {@code false} (see
     * {@code Gemma4Model}, {@code Qwen3MoeModel}, {@code MixtralModel}) -- step 4 plan Section 6.
     */
    @Override
    protected boolean supportsLoraHotSwap() {
        return true;
    }

    @Override
    protected TransformerBlock[] loadTransformerBlockWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        TransformerBlock[] transformerBlocks = new TransformerBlock[config.numberOfLayers];
        IntStream.range(0, config.numberOfLayers).parallel().forEach(i -> {
            int relativeLayer = i;
            String base = "model.layers." + i + ".";
            String prefix = base + "self_attn.";
            String qName = prefix + "q_proj.weight";
            String kName = prefix + "k_proj.weight";
            String vName = prefix + "v_proj.weight";
            String oName = prefix + "o_proj.weight";
            AbstractTensor qWeight = quantize(weights.load(qName), qType);
            AbstractTensor kWeight = quantize(weights.load(kName), qType);
            AbstractTensor vWeight = quantize(weights.load(vName), qType);
            AbstractTensor oWeight = quantize(weights.load(oName), qType);
            registerModelLineageTensor(qName, qWeight);
            registerModelLineageTensor(kName, kWeight);
            registerModelLineageTensor(vName, vWeight);
            registerModelLineageTensor(oName, oWeight);
            CausalSelfAttention attention = new CausalSelfAttention(
                    this,
                    relativeLayer,
                    qWeight,
                    kWeight,
                    vWeight,
                    oWeight,
                    configurableTensorProvider,
                    metricRegistry,
                    qName, kName, vName, oName
            );

            prefix = base + "mlp.";
            String gateName = prefix + "gate_proj.weight";
            String downName = prefix + "down_proj.weight";
            String upName = prefix + "up_proj.weight";
            AbstractTensor gateWeight = quantize(weights.load(gateName), qType);
            AbstractTensor downWeight = quantize(weights.load(downName), qType);
            AbstractTensor upWeight = quantize(weights.load(upName), qType);
            registerModelLineageTensor(gateName, gateWeight);
            registerModelLineageTensor(downName, downWeight);
            registerModelLineageTensor(upName, upWeight);
            MLPBlock mlp = new MLPBlock(
                    this,
                    config.activationFunction,
                    gateWeight, // w1
                    downWeight, // w2
                    upWeight,
                    configurableTensorProvider,
                    gateName, upName, downName
            ); // w3

            AbstractTensor inputNormWeight = quantize(weights.load(base + "input_layernorm.weight"), qType);
            AbstractTensor postAttentionNormWeight = quantize(weights.load(base + "post_attention_layernorm.weight"), qType);
            registerModelLineageTensor(base + "input_layernorm.weight", inputNormWeight);
            registerModelLineageTensor(base + "post_attention_layernorm.weight", postAttentionNormWeight);

            transformerBlocks[relativeLayer] = new TransformerBlock(
                    this,
                    relativeLayer,
                    new RmsNorm(this, inputNormWeight, metricRegistry),
                    attention,
                    new RmsNorm(this, postAttentionNormWeight, metricRegistry),
                    mlp,
                    configurableTensorProvider
            );
        });
        return transformerBlocks;
    }

    @Override
    protected SampleOutput loadOutputWeights() {
        DType qType = modelQType.orElse(this.modelDType);
        LOGGER.debug("loading output norm weight=model.norm.weight target_dtype={}", qType);
        AbstractTensor outputNormWeight = quantize(weights.load("model.norm.weight"), qType);
        registerModelLineageTensor("model.norm.weight", outputNormWeight);
        final LayerNorm outputLayerNorm = new RmsNorm(this, outputNormWeight, metricRegistry);
        DType outputHeadDType = outputHeadQuantization.orElse(workingDType);
        boolean forceOutputHeadQuantization = outputHeadQuantization.isPresent();
        // Some llama models don't have a classification head
        boolean hasLmHead = weights.isWeightPresent("lm_head.weight");
        LOGGER.debug("loading output logits weight={} target_dtype={} force_quantization={}",
                hasLmHead ? "lm_head.weight" : "model.embed_tokens.weight", outputHeadDType, forceOutputHeadQuantization);
        AbstractTensor classificationWeights = weights.isWeightPresent("lm_head.weight")
                ? io.teknek.deliverance.tensor.AbstractTensorUtils.quantize(weights.load("lm_head.weight"), outputHeadDType,
                forceOutputHeadQuantization)
                : io.teknek.deliverance.tensor.AbstractTensorUtils.quantize(
                        embedTokenWeights == null ? weights.load("model.embed_tokens.weight") : embedTokenWeights,
                        outputHeadDType,
                        forceOutputHeadQuantization);
        registerModelLineageTensor(hasLmHead ? "lm_head.weight" : "model.embed_tokens.weight#output_head", classificationWeights);
        LOGGER.debug("loaded output logits shape={} dtype={}", classificationWeights.shape(), classificationWeights.dType());
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
