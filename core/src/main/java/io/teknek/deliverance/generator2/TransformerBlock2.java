package io.teknek.deliverance.generator2;

import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** TensorRef-native transformer block. */
public final class TransformerBlock2 {
    private final AbstractModel model;
    private final int layerIndex;
    private final RmsNorm2 preAttentionNorm;
    private final Qwen3KvCacheSelfAttention2 attention;
    private final RmsNorm2 postAttentionNorm;
    private final MLPBlock2 mlp;
    private final CompositeOps compositeOps;

    public TransformerBlock2(AbstractModel model, int layerIndex, RmsNorm2 preAttentionNorm,
            Qwen3KvCacheSelfAttention2 attention, RmsNorm2 postAttentionNorm, MLPBlock2 mlp) {
        this.model = java.util.Objects.requireNonNull(model, "model");
        this.layerIndex = layerIndex;
        this.preAttentionNorm = java.util.Objects.requireNonNull(preAttentionNorm, "preAttentionNorm");
        this.attention = java.util.Objects.requireNonNull(attention, "attention");
        this.postAttentionNorm = java.util.Objects.requireNonNull(postAttentionNorm, "postAttentionNorm");
        this.mlp = java.util.Objects.requireNonNull(mlp, "mlp");
        this.compositeOps = new CompositeOps(model.getLighter(), model.getMetricRegistry());
    }

    public TensorRef forward(TensorRef embedding, int startPosition, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase) {
        TensorRef normalized = preAttentionNorm.forward(embedding);
        TensorRef attentionOutput = attention.forward(normalized, startPosition, kvSession, tensorReducer, phase);
        normalized.close();
        compositeOps.addResidual(attentionOutput, embedding, model.getConfig().residualMultiplier == null
                ? 1.0f : model.getConfig().residualMultiplier);

        TensorRef residualBase = attentionOutput;
        TensorRef postAttention = postAttentionNorm.forward(residualBase);
        TensorRef feedForward = mlp.forward(postAttention);
        postAttention.close();
        compositeOps.addResidual(feedForward, residualBase, model.getConfig().residualMultiplier == null
                ? 1.0f : model.getConfig().residualMultiplier);
        residualBase.close();
        return feedForward;
    }
}
