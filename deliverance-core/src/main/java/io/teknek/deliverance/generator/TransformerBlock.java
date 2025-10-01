package io.teknek.deliverance.generator;




import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.operations.TensorOperationsProvider;
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

    public TransformerBlock(
            AbstractModel model,
            int layerIndex,
            LayerNorm preAttentionNorm,
            CausalSelfAttention attention,
            LayerNorm postAttentionNorm,
            FeedForward ffBlock
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
                Optional.empty()
        );
    }

    public TransformerBlock(
            AbstractModel model,
            int layerIndex,
            CausalSelfAttention attention,
            LayerNorm postAttentionNorm,
            FeedForward ffBlock,
            LayerNorm postFFNorm
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
                Optional.of(postFFNorm)
        );
    }

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
    }

    public TransformerBlock(
            AbstractModel model,
            int layerIndex,
            LayerNorm preAttentionNorm,
            CausalSelfAttention attention,
            LayerNorm postAttentionNorm,
            LayerNorm preFFNorm,
            FeedForward ffBlock,
            LayerNorm postFFNorm
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
                Optional.empty()
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
            Optional<LayerNorm> preResponseNorm
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

        debug("input_emb", embedding, layerIndex);

        AbstractTensor lnemb = preAttentionNorm.map(ln -> ln.forward(embedding)).orElse(embedding);

        debug("ln_emb", lnemb, layerIndex);

        AbstractTensor postAttention;
        try (AbstractTensor qlnemb = model.maybeQuantize(lnemb)) {
            postAttention = attention.forward(qlnemb, position, kvBuffer, tensorReducer);
        }

        debug("post_attn", postAttention, layerIndex);
        AbstractTensor lnattn = maybeApplyNorm(postAttention, postAttentionNorm);

        debug("post_attn_norm", lnattn, layerIndex);

        // residual connection
        if (model.getConfig().residualMultiplier != null) {
            TensorOperationsProvider.get().scale(model.getConfig().residualMultiplier, lnattn, 0, model.getConfig().embeddingLength);
        }
        TensorOperationsProvider.get().accumulate(lnattn, embedding, 0, model.getConfig().embeddingLength);

        AbstractTensor lnpreFF = preFFNorm.map(ln -> ln.forward(lnattn)).orElse(lnattn);

        debug("pre_ff_norm", lnpreFF, layerIndex);

        AbstractTensor postFF;
        try (AbstractTensor qlnemb2 = model.maybeQuantize(lnpreFF)) {
            postFF = ffBlock.forward(qlnemb2, tensorReducer);
            debug("post_ff", postFF, layerIndex);
        }

        AbstractTensor lnpostFF = maybeApplyNorm(postFF, postFFNorm);

        // residual connection
        if (model.getConfig().residualMultiplier != null) {
            TensorOperationsProvider.get().scale(model.getConfig().residualMultiplier, lnpostFF, 0, model.getConfig().embeddingLength);
        }
        TensorOperationsProvider.get().accumulate(lnpostFF, lnattn, 0, model.getConfig().embeddingLength);

        debug("post_ff_res", lnpostFF, layerIndex);

        // Release any tmp buffers (embedding is released by caller)
        if (lnemb != embedding) lnemb.close();
        if (lnattn != postAttention) lnattn.close();
        else postAttention.close();
        if (lnpreFF != lnattn) lnpreFF.close();
        else lnattn.close();

        return maybeApplyNorm(lnpostFF, preResponseNorm);
    }

    private AbstractTensor maybeApplyNorm(AbstractTensor tensor, Optional<LayerNorm> norm) {
        return norm.map(ln -> {
            AbstractTensor o = ln.forward(tensor);
            tensor.close();
            return o;
        }).orElse(tensor);
    }
}
