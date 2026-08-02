package io.teknek.deliverance.generator;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

import com.codahale.metrics.Timer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensorlib.TensorPlan;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static io.teknek.deliverance.tensor.DebugSupport.debug;

public class TransformerBlock {

    private static final Logger logger = LoggerFactory.getLogger(TransformerBlock.class);

    private final AbstractModel model;
    final int layerIndex;
    final Optional<LayerNorm> preAttentionNorm;
    final SelfAttention attention;
    final Optional<LayerNorm> postAttentionNorm; // After attention, before the residual connection
    final Optional<LayerNorm> preFFNorm; // After residual connection, before the FF
    final FeedForward ffBlock;
    final Optional<LayerNorm> postFFNorm; // After FF, before the residual connection
    final Optional<LayerNorm> preResponseNorm; // After the residual connection
    final ConfigurableTensorProvider configurableTensorProvider;

    public TransformerBlock(AbstractModel model, int layerIndex, LayerNorm preAttentionNorm,
            SelfAttention attention, LayerNorm postAttentionNorm, FeedForward ffBlock,
            ConfigurableTensorProvider configurableTensorProvider) {
        this(model, layerIndex, Optional.of(preAttentionNorm), attention, Optional.empty(),
                Optional.of(postAttentionNorm), ffBlock, Optional.empty(), Optional.empty(),
                configurableTensorProvider);
    }


    public TransformerBlock(
            AbstractModel model,
            int layerIndex,
            SelfAttention attention,
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
            SelfAttention attention,
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
            SelfAttention attention,
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
            SelfAttention attention,
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

    public AbstractTensor forward(AbstractTensor embedding, int position, KvBufferCache.KvBuffer kvBuffer,
            ForwardPhase phase) {
        return forward(embedding, position, kvBuffer, Optional.empty(), phase);
    }

    public AbstractTensor forward(
            AbstractTensor embedding,
            int position,
            KvBufferCache.KvBuffer kvBuffer,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer
    ) {
        return forward(embedding, position, kvBuffer, tensorReducer, ForwardPhase.DECODE);
    }

    public AbstractTensor forward(
            AbstractTensor embedding,
            int position,
            KvBufferCache.KvBuffer kvBuffer,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer,
            ForwardPhase phase
    ) {
        Timer timer = InferenceProfiler.timer(model.getMetricRegistry(), "transformerblock.forward");
        try (Timer.Context ignored = timer.time()) {
        AbstractTensor lnemb = preAttentionNorm.map(ln -> ln.forward(embedding)).orElse(embedding);
        AbstractTensor postAttention;
        try (AbstractTensor qlnemb = model.maybeQuantize(lnemb)) {
            postAttention = attention.forward(qlnemb, position, kvBuffer, tensorReducer, phase);
        }
        AbstractTensor lnattn = maybeApplyNorm(postAttention, postAttentionNorm);
        applyResidual(lnattn, embedding, "post_attention_residual");
        model.emitLayerDebug(layerIndex, "post_attention_residual", lnattn);

        AbstractTensor lnpreFF = preFFNorm.map(ln -> ln.forward(lnattn)).orElse(lnattn);
        AbstractTensor postFF;
        try (AbstractTensor qlnemb2 = model.maybeQuantize(lnpreFF)) {
            postFF = ffBlock.forward(qlnemb2, tensorReducer, phase);
        }

        AbstractTensor lnpostFF = maybeApplyNorm(postFF, postFFNorm);

        applyResidual(lnpostFF, lnattn, "post_ff_residual");
        model.emitLayerDebug(layerIndex, "post_ff_residual", lnpostFF);

        // Release any tmp buffers (embedding is released by caller)
        if (lnemb != embedding) lnemb.close();
        if (lnattn != postAttention) lnattn.close();
        else postAttention.close();
        if (lnpreFF != lnattn) lnpreFF.close();
        else lnattn.close();

        return maybeApplyNorm(lnpostFF, preResponseNorm);
        }
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

    private void applyResidual(AbstractTensor target, AbstractTensor residual, String name) {
        TensorPlan plan = TensorPlanSupport.plan(model, configurableTensorProvider.get());
        plan.fuse(name, target.shape())
                .write("target", plan.mutable("target", target))
                .read("residual", plan.input("residual", residual))
                .map(name + " = residual(target, residual)", TensorPlan.TensorOp.CUSTOM,
                        (ctx, offset, length) -> applyResidualRange(ctx.tensor("target"), ctx.tensor("residual"),
                                model.getConfig().residualMultiplier, offset, length))
                .tensor()
                .materialize();
    }

    static void applyResidualRange(AbstractTensor target, AbstractTensor residual, Float multiplier, long offset,
            long length) {
        int columns = (int) target.shape().last();
        long end = offset + length;
        for (long index = offset; index < end; index++) {
            int row = (int) (index / columns);
            int column = (int) (index % columns);
            float value = target.get(row, column);
            if (multiplier != null) {
                value *= multiplier;
            }
            target.set(value + residual.get(row, column), row, column);
        }
    }
}
