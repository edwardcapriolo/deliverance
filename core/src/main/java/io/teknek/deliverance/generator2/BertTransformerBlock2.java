package io.teknek.deliverance.generator2;

import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** TensorRef BERT block using the post-residual LayerNorm ordering from BERT. */
public final class BertTransformerBlock2 extends TransformerBlock2 {
    private final AbstractModel model;
    private final SelfAttention2 attention;
    private final LayerNorm2 attentionNorm;
    private final FeedForward2 feedForward;
    private final LayerNorm2 outputNorm;
    private final CompositeOps compositeOps;

    public BertTransformerBlock2(AbstractModel model, int layerIndex, SelfAttention2 attention,
            LayerNorm2 attentionNorm, FeedForward2 feedForward, LayerNorm2 outputNorm) {
        super(model, layerIndex, Optional.empty(), attention, Optional.empty(), Optional.empty(), feedForward,
                Optional.empty(), Optional.empty(), model.getConfigurableTensorProvider());
        this.model = model;
        this.attention = attention;
        this.attentionNorm = attentionNorm;
        this.feedForward = feedForward;
        this.outputNorm = outputNorm;
        this.compositeOps = new CompositeOps(model.getLighter(), model.getMetricRegistry());
    }

    @Override
    public TensorRef forward(TensorRef embedding, int startPosition, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase) {
        return forward(embedding, startPosition, kvSession, tensorReducer, phase,
                1, (int) embedding.shape().first(), null);
    }

    @Override
    public TensorRef forward(TensorRef embedding, int startPosition, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase, int batchSize,
            int sequenceLength, int[] attentionMask) {
        TensorRef postAttention;
        try (AbstractModel.TensorRefLease input = model.maybeQuantizeReadOnly(embedding,
                "berttransformerblock2.maybe_quantize.attention")) {
            postAttention = attention.forward(input.tensor(), startPosition, kvSession, tensorReducer, phase,
                    batchSize, sequenceLength, attentionMask);
        }
        compositeOps.addResidual(postAttention, embedding, model.getConfig().residualMultiplier == null
                ? 1.0f : model.getConfig().residualMultiplier);
        TensorRef attentionOutput = attentionNorm.forward(postAttention);
        postAttention.close();

        TensorRef postFeedForward;
        try (AbstractModel.TensorRefLease input = model.maybeQuantizeReadOnly(attentionOutput,
                "berttransformerblock2.maybe_quantize.feed_forward")) {
            postFeedForward = feedForward.forward(input.tensor(), tensorReducer, phase);
        }
        compositeOps.addResidual(postFeedForward, attentionOutput, model.getConfig().residualMultiplier == null
                ? 1.0f : model.getConfig().residualMultiplier);
        TensorRef output = outputNorm.forward(postFeedForward);
        postFeedForward.close();
        attentionOutput.close();
        return output;
    }
}
