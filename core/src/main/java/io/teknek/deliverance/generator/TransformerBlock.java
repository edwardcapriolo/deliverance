package io.teknek.deliverance.generator;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static io.teknek.deliverance.tensor.DebugSupport.debug;

public class TransformerBlock {

    private static final Logger logger = LoggerFactory.getLogger(TransformerBlock.class);

    private final AbstractModel model;
    final int layerIndex;
    final Optional<LayerNorm> preAttentionNorm;
    final CausalSelfAttention attention;
    final Optional<LayerNorm> postAttentionNorm; // After attention, before the residual connection
    final Optional<LayerNorm> preFFNorm; // After residual connection, before the FF
    final FeedForward ffBlock;
    final Optional<LayerNorm> postFFNorm; // After FF, before the residual connection
    final Optional<LayerNorm> preResponseNorm; // After the residual connection
    final ConfigurableTensorProvider configurableTensorProvider;

    public TransformerBlock(AbstractModel model, int layerIndex, LayerNorm preAttentionNorm,
            CausalSelfAttention attention, LayerNorm postAttentionNorm, FeedForward ffBlock,
            ConfigurableTensorProvider configurableTensorProvider) {
        this(model, layerIndex, Optional.of(preAttentionNorm), attention, Optional.empty(),
                Optional.of(postAttentionNorm), ffBlock, Optional.empty(), Optional.empty(),
                configurableTensorProvider);
    }


    public TransformerBlock(
            AbstractModel model,
            int layerIndex,
            CausalSelfAttention attention,
            LayerNorm postAttentionNorm,
            FeedForward ffBlock,
            LayerNorm postFFNorm,
            ConfigurableTensorProvider configurableTensorProvider
    ) {
        this(
                model,
                layerIndex,
                Optional.empty(),
                attention,
                Optional.empty(),
                Optional.of(postAttentionNorm),
                ffBlock,
                Optional.empty(),
                Optional.of(postFFNorm),
                configurableTensorProvider
        );
    }

/*
    public TransformerBlock(
            AbstractModel model,
            int layerIndex,
            LayerNorm preAttentionNorm,
            CausalSelfAttention attention,
            LayerNorm postAttentionNorm,
            FeedForward ffBlock,
            LayerNorm postFFNorm
    ) {
        this(
                model,
                layerIndex,
                Optional.of(preAttentionNorm),
                attention,
                Optional.empty(),
                Optional.of(postAttentionNorm),
                ffBlock,
                Optional.empty(),
                Optional.of(postFFNorm)
        );
    }*/




    public TransformerBlock(
            AbstractModel model,
            int layerIndex,
            LayerNorm preAttentionNorm,
            CausalSelfAttention attention,
            LayerNorm postAttentionNorm,
            LayerNorm preFFNorm,
            FeedForward ffBlock,
            LayerNorm postFFNorm,
            ConfigurableTensorProvider configurableTensorProvider
    ) {
        this(
                model,
                layerIndex,
                Optional.of(preAttentionNorm),
                attention,
                Optional.of(postAttentionNorm),
                Optional.of(preFFNorm),
                ffBlock,
                Optional.of(postFFNorm),
                Optional.empty(),
                configurableTensorProvider
        );
    }

    public TransformerBlock(
            AbstractModel model,
            int layerIndex,
            Optional<LayerNorm> preAttentionNorm,
            CausalSelfAttention attention,
            Optional<LayerNorm> postAttentionNorm,
            Optional<LayerNorm> preFFNorm,
            FeedForward ffBlock,
            Optional<LayerNorm> postFFNorm,
            Optional<LayerNorm> preResponseNorm,
            ConfigurableTensorProvider configurableTensorProvider
    ) {

        this.model = model;
        this.layerIndex = layerIndex;
        this.preAttentionNorm = preAttentionNorm;
        this.attention = attention;
        this.postAttentionNorm = postAttentionNorm;
        this.preFFNorm = preFFNorm;
        this.ffBlock = ffBlock;
        this.postFFNorm = postFFNorm;
        this.preResponseNorm = preResponseNorm;
        this.configurableTensorProvider = configurableTensorProvider;
    }

    public AbstractTensor forward(AbstractTensor embedding, int position, KvBufferCache.KvBuffer kvBuffer) {
        return forward(embedding, position, kvBuffer, Optional.empty());
    }

    public AbstractTensor forward(
            AbstractTensor embedding,
            int position,
            KvBufferCache.KvBuffer kvBuffer,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer
    ) {
        AbstractTensor lnemb = preAttentionNorm.map(ln -> ln.forward(embedding)).orElse(embedding);
        AbstractTensor postAttention;
        try (AbstractTensor qlnemb = model.maybeQuantize(lnemb)) {
            postAttention = attention.forward(qlnemb, position, kvBuffer, tensorReducer);
        }
        AbstractTensor lnattn = maybeApplyNorm(postAttention, postAttentionNorm);
        // residual connection
        if (model.getConfig().residualMultiplier != null) {
            configurableTensorProvider.get().scale(model.getConfig().residualMultiplier, lnattn, 0, model.getConfig().embeddingLength);
        }
        configurableTensorProvider.get().accumulate(lnattn, embedding, 0, model.getConfig().embeddingLength);
        model.emitLayerDebug(layerIndex, "post_attention_residual", lnattn);

        AbstractTensor lnpreFF = preFFNorm.map(ln -> ln.forward(lnattn)).orElse(lnattn);
        AbstractTensor postFF;
        try (AbstractTensor qlnemb2 = model.maybeQuantize(lnpreFF)) {
            postFF = ffBlock.forward(qlnemb2, tensorReducer);
        }

        AbstractTensor lnpostFF = maybeApplyNorm(postFF, postFFNorm);

        // residual connection
        if (model.getConfig().residualMultiplier != null) {
            configurableTensorProvider.get().scale(model.getConfig().residualMultiplier, lnpostFF, 0, model.getConfig().embeddingLength);
        }
        configurableTensorProvider.get().accumulate(lnpostFF, lnattn, 0, model.getConfig().embeddingLength);
        model.emitLayerDebug(layerIndex, "post_ff_residual", lnpostFF);

        // Release any tmp buffers (embedding is released by caller)
        if (lnemb != embedding) lnemb.close();
        if (lnattn != postAttention) lnattn.close();
        else postAttention.close();
        if (lnpreFF != lnattn) lnpreFF.close();
        else lnattn.close();

        return maybeApplyNorm(lnpostFF, preResponseNorm);
    }

    /**
     *
     * @param tensor
     * @param norm
     * @return if norm is supplied call norm.forward on tensor and return new result otherwise return tensor,
     */
    private AbstractTensor maybeApplyNorm(AbstractTensor tensor, Optional<LayerNorm> norm) {
        return norm.map(ln -> {
            AbstractTensor o = ln.forward(tensor);
            tensor.close();
            return o;
        }).orElse(tensor);
    }
}
