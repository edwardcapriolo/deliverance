package io.teknek.deliverance.model.llama;

import io.dropwizard.metrics5.MetricRegistry;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.*;
import io.teknek.deliverance.generator2.LayerNorm2;
import io.teknek.deliverance.generator2.CausalSelfAttention2;
import io.teknek.deliverance.generator2.MLPBlock2;
import io.teknek.deliverance.generator2.RmsNorm2;
import io.teknek.deliverance.generator2.SampleOutputRef;
import io.teknek.deliverance.generator2.TransformerBlock2;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.tensorparallel.TensorParallelCollectives;
import io.teknek.deliverance.model.tensorparallel.TensorParallelContext;
import io.teknek.deliverance.safetensors.Config;
import io.teknek.deliverance.safetensors.WeightLoader;
import io.teknek.deliverance.tensor.AbstractTensor;

import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.toolcallparser.ToolCallParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.stream.IntStream;

import static io.teknek.deliverance.tensor.AbstractTensorUtils.quantize;

public class LlamaModel extends AbstractModel {
    private static final Logger LOGGER = LoggerFactory.getLogger(LlamaModel.class);

    private volatile AbstractTensor embedTokenWeights;
    private volatile TensorRef embedTokenWeightsRef;
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
        if (getClass() == LlamaModel.class) {
            embedTokenWeightsRef = registerModelTensorRef(weights.loadRef("model.embed_tokens.weight"));
            return new EmbedInput(this) {
                @Override
                public TensorRef inputTokenToEmbedding(int inputToken, int position) {
                    try (TensorRef row = embedTokenWeightsRef.slice(inputToken)) {
                        TensorRef embedding = makeDenseTensorRef(1, config.embeddingLength);
                        try {
                            if (row.dType() != workingDType) {
                                lighter.reshape(row, embedding);
                            } else {
                                lighter.copy(row, 0, embedding, 0, config.embeddingLength);
                            }
                            return embedding;
                        } catch (RuntimeException | Error e) {
                            embedding.close();
                            throw e;
                        }
                    }
                }
            };
        }

        //TODO resolvethis
        // Don't quantize this, it's used for the embedding layer
        // but we ae calling quantize in the if?
        if (embedTokenWeights == null) {
            //embedTokenWeights = weights.load("model.embed_tokens.weight").quantize(workingDType);
            LOGGER.debug("loading input embeddings weight=model.embed_tokens.weight target_dtype={}", workingDType);
            AbstractTensor loadedEmbeddings = weights.load("model.embed_tokens.weight");
            // Concrete Llama TensorRef execution keeps the vocabulary matrix compressed and converts only
            // the rows requested by the current prompt. Expanding the complete Q4 matrix duplicates the
            // TensorRef model weights and can exhaust direct memory during model initialization.
            embedTokenWeights = getClass() == LlamaModel.class
                    ? loadedEmbeddings : quantize(loadedEmbeddings, workingDType);
            LOGGER.debug("loaded input embeddings shape={} dtype={}", embedTokenWeights.shape(), embedTokenWeights.dType());
            registerModelLineageTensor("model.embed_tokens.weight", embedTokenWeights);
            configurableTensorProvider.get().registerModelTensor(embedTokenWeights);
        }

        return new EmbedInput(this) {
            @Override
            //TODO The second argument position was  double check that this is propper
            public TensorRef inputTokenToEmbedding(int inputToken, int unused) {
                if (embedTokenWeights.dType() == DType.BF16) {
                    // Handle old style model with BF16 embeddings
                    AbstractTensor embedding = makeDenseTensor(1, config.embeddingLength);
                    AbstractTensor at = embedTokenWeights.slice(true, inputToken);
                    if (embedTokenWeights.dType() != embedding.dType()) {
                        at = configurableTensorProvider.get().quantize(at, embedding.dType(), 0, config.embeddingLength);
                    }
                    embedding.copyFrom(at, 0, 0, config.embeddingLength);
                    return TensorRef.owned(embedding);
                } else {
                    AbstractTensor at = embedTokenWeights.slice(true, inputToken);
                    AbstractTensor embedding = parent.getTensorAllocator().getDirty(at.dType(), at.shape());
                    embedding.copyFrom(at, 0, 0, config.embeddingLength);
                    return TensorRef.owned(embedding);
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

    /** Enables the TensorRef path only for the concrete Llama family, not its subclasses. */
    @Override
    protected boolean usesTensorRefExecution() {
        return getClass() == LlamaModel.class;
    }

    @Override
    public boolean usesKvCache2Generation() {
        return getClass() == LlamaModel.class;
    }

    @Override
    protected TransformerBlock2[] loadTransformerBlockWeights2() {
        if (getClass() != LlamaModel.class) {
            throw new UnsupportedOperationException("TensorRef Llama blocks are concrete-Llama only");
        }
        if (tensorParallelContext.enabled()) {
            throw new UnsupportedOperationException("TensorRef Llama tensor parallelism is not ported");
        }
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
            MLPBlock2 mlp = new MLPBlock2(this,
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(gateName, weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(upName, weights::loadRef, qType)),
                    registerModelTensorRef(loadAndMaybeQuantizedExcluding1DTensors(downName, weights::loadRef, qType)),
                    lighter, gateName, upName, downName);

            String inputNormName = base + "input_layernorm.weight";
            String postAttentionNormName = base + "post_attention_layernorm.weight";
            blocks[i] = new TransformerBlock2(this, i,
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(
                            loadAndMaybeQuantizedExcluding1DTensors(inputNormName, weights::loadRef, qType)), 0.0f)),
                    attention,
                    Optional.empty(),
                    Optional.of(new RmsNorm2(this, registerModelTensorRef(
                            loadAndMaybeQuantizedExcluding1DTensors(postAttentionNormName, weights::loadRef, qType)), 0.0f)),
                    mlp, Optional.empty(), Optional.empty(), configurableTensorProvider);
        });
        return blocks;
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

    @Override
    protected SampleOutputRef loadOutputWeightsRef() {
        DType qType = modelQType.orElse(this.modelDType);
        TensorRef norm = weights.loadRef("model.norm.weight");
        TensorRef head = weights.isWeightPresent("lm_head.weight")
                ? weights.loadRef("lm_head.weight")
                : getClass() == LlamaModel.class && embedTokenWeightsRef != null
                        ? embedTokenWeightsRef : weights.loadRef("model.embed_tokens.weight");
        if (lighter.shouldQuantizeForEfficiency(norm, qType)) {
            TensorRef quantized = lighter.reshape(norm, qType);
            norm.close();
            norm = quantized;
        }
        if (outputHeadQuantization.isPresent() && lighter.shouldQuantizeForEfficiency(head,
                outputHeadQuantization.get())) {
            TensorRef quantized = lighter.reshape(head, outputHeadQuantization.get());
            if (head != embedTokenWeightsRef) {
                head.close();
            }
            head = quantized;
        }
        norm = registerModelTensorRef(norm);
        head = registerModelTensorRef(head);
        TensorRef finalNorm = norm;
        TensorRef finalHead = head;
        return new SampleOutputRef() {
            @Override
            public LayerNorm2 outputLayerNorm() {
                return new RmsNorm2(LlamaModel.this, finalNorm, 0.0f);
            }

            @Override
            public TensorRef outputLogitsWeights() {
                return finalHead;
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
