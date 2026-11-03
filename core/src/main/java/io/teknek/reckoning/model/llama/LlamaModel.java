package io.teknek.reckoning.model.llama;

import io.dropwizard.metrics5.MetricRegistry;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.teknek.reckoning.DType;
import io.teknek.reckoning.generator.*;
import io.teknek.reckoning.generator2.LayerNorm2;
import io.teknek.reckoning.generator2.CausalSelfAttention2;
import io.teknek.reckoning.generator2.MLPBlock2;
import io.teknek.reckoning.generator2.RmsNorm2;
import io.teknek.reckoning.generator2.SampleOutputRef;
import io.teknek.reckoning.generator2.TransformerBlock2;
import io.teknek.reckoning.grace.PreTrainedTokenizer;
import io.teknek.reckoning.math.WrappedForkJoinPool;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.tensorparallel.TensorParallelCollectives;
import io.teknek.reckoning.model.tensorparallel.TensorParallelContext;
import io.teknek.reckoning.safetensors.Config;
import io.teknek.reckoning.safetensors.WeightLoader;
import io.teknek.reckoning.tensor.AbstractTensor;

import io.teknek.reckoning.tensor.KvBufferCacheSettings;
import io.teknek.reckoning.tensor2.TensorRef;
import io.teknek.reckoning.tensor2.TensorRefBackedTensor;
import io.teknek.reckoning.tensor2.Lighter;
import io.teknek.reckoning.toolcallparser.ToolCallParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.stream.IntStream;

import static io.teknek.reckoning.tensor.AbstractTensorUtils.quantize;

public class LlamaModel extends AbstractModel {
    private static final Logger LOGGER = LoggerFactory.getLogger(LlamaModel.class);

    private volatile AbstractTensor embedTokenWeights;
    private volatile TensorRef embedTokenWeightsRef;

    public LlamaModel(InferenceType inferenceType, Config c, WeightLoader w, PreTrainedTokenizer t,
            DType workingMemoryDType, DType workingQType, Optional<DType> modelQType,
            MetricRegistry metricRegistry, KvBufferCacheSettings kvBufferCacheSettings,
            ToolCallParser toolCallParser, WrappedForkJoinPool pool, TensorParallelContext tensorParallelContext,
            TensorParallelCollectives tensorParallelCollectives, Optional<DType> outputHeadQuantization,
            Lighter lighter) {
        super(inferenceType, c, w, t, workingMemoryDType, workingQType, modelQType, metricRegistry,
                kvBufferCacheSettings, toolCallParser, pool, tensorParallelContext, tensorParallelCollectives,
                outputHeadQuantization, lighter);
    }

    @Override
    protected EmbedInput loadInputWeights() {
        if (usesTensorRefExecution()) {
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
        }

        return new EmbedInput(this) {
            @Override
            //TODO The second argument position was  double check that this is propper
            public TensorRef inputTokenToEmbedding(int inputToken, int unused) {
                if (embedTokenWeights.dType() == DType.BF16) {
                    // Handle old style model with BF16 embeddings
                    AbstractTensor embedding = new TensorRefBackedTensor(makeDenseTensorRef(1, config.embeddingLength));
                    AbstractTensor at = embedTokenWeights.slice(true, inputToken);
                    if (embedTokenWeights.dType() != embedding.dType()) {
                        at = quantize(at, embedding.dType());
                    }
                    embedding.copyFrom(at, 0, 0, config.embeddingLength);
                    return TensorRef.owned(embedding);
                } else {
                    AbstractTensor at = embedTokenWeights.slice(true, inputToken);
                    AbstractTensor embedding = new TensorRefBackedTensor(parent.getLighter().allocateDirty(at.dType(), at.shape()));
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


    /** Enables the TensorRef path only for the concrete Llama family, not its subclasses. */
    @Override
    protected boolean usesTensorRefExecution() {
        return getClass() == LlamaModel.class || getClass() == io.teknek.reckoning.model.mistral.MistralModel.class;
    }

    @Override
    public boolean usesKvCache2Generation() {
        return getClass() == LlamaModel.class;
    }

    @Override
    protected TransformerBlock2[] loadTransformerBlockWeights2() {
        if (getClass() != LlamaModel.class && getClass() != io.teknek.reckoning.model.mistral.MistralModel.class) {
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
                     mlp, Optional.empty(), Optional.empty());
        });
        return blocks;
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
        return quantize(t, workingQType);
    }

}
