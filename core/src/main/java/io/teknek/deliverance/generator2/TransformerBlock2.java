package io.teknek.deliverance.generator2;

import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** TensorRef-native transformer block. */
public class TransformerBlock2 {
    private final AbstractModel model;
    final int layerIndex;
    final Optional<LayerNorm2> preAttentionNorm;
    final SelfAttention2 attention;
    final Optional<LayerNorm2> postAttentionNorm;
    final Optional<LayerNorm2> preFFNorm;
    final FeedForward2 ffBlock;
    final Optional<LayerNorm2> postFFNorm;
    final Optional<LayerNorm2> preResponseNorm;
    final ConfigurableTensorProvider configurableTensorProvider;
    private final CompositeOps compositeOps;

    public TransformerBlock2(AbstractModel model, int layerIndex, LayerNorm2 preAttentionNorm,
            Qwen3KvCacheSelfAttention2 attention, LayerNorm2 postAttentionNorm, FeedForward2 mlp) {
        this(model, layerIndex, Optional.of(preAttentionNorm), attention, Optional.empty(),
                Optional.of(postAttentionNorm), mlp, Optional.empty(), Optional.empty(),
                model.getConfigurableTensorProvider());
    }

    public TransformerBlock2(
            AbstractModel model,
            int layerIndex,
            Optional<LayerNorm2> preAttentionNorm,
            SelfAttention2 attention,
            Optional<LayerNorm2> postAttentionNorm,
            Optional<LayerNorm2> preFFNorm,
            FeedForward2 ffBlock,
            Optional<LayerNorm2> postFFNorm,
            Optional<LayerNorm2> preResponseNorm,
            ConfigurableTensorProvider configurableTensorProvider
    ) {
        this.model = java.util.Objects.requireNonNull(model, "model");
        this.layerIndex = layerIndex;
        this.preAttentionNorm = java.util.Objects.requireNonNull(preAttentionNorm, "preAttentionNorm");
        this.attention = java.util.Objects.requireNonNull(attention, "attention");
        this.postAttentionNorm = java.util.Objects.requireNonNull(postAttentionNorm, "postAttentionNorm");
        this.preFFNorm = java.util.Objects.requireNonNull(preFFNorm, "preFFNorm");
        this.ffBlock = java.util.Objects.requireNonNull(ffBlock, "ffBlock");
        this.postFFNorm = java.util.Objects.requireNonNull(postFFNorm, "postFFNorm");
        this.preResponseNorm = java.util.Objects.requireNonNull(preResponseNorm, "preResponseNorm");
        this.configurableTensorProvider = configurableTensorProvider;
        this.compositeOps = new CompositeOps(model.getLighter(), model.getMetricRegistry());
    }

    public TensorRef forward(TensorRef embedding, int startPosition, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase) {
        return forward(embedding, startPosition, kvSession, tensorReducer, phase,
                1, (int) embedding.shape().first(), null);
    }

    public TensorRef forward(TensorRef embedding, int startPosition, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase, int batchSize,
            int sequenceLength, int[] attentionMask) {
        Timer timer = InferenceProfiler.timer(model.getMetricRegistry(), "transformerblock.forward");
        try (Timer.Context ignored = timer.time()) {
            TensorRef lnemb = preAttentionNorm.map(ln -> ln.forward(embedding)).orElse(embedding);
            TensorRef postAttention;
            try (AbstractModel.TensorRefLease qlnemb = preAttentionProjectionInput(lnemb)) {
                postAttention = attention.forward(qlnemb.tensor(), startPosition, kvSession, tensorReducer, phase,
                        batchSize, sequenceLength, attentionMask);
            }
            TensorRef lnattn = maybeApplyNorm(postAttention, postAttentionNorm);
            applyResidual(lnattn, embedding, "post_attention_residual");
            model.emitLayerDebug(layerIndex, "post_attention_residual", lnattn);

            TensorRef lnpreFF = preFFNorm.map(ln -> ln.forward(lnattn)).orElse(lnattn);
            TensorRef postFF;
            try (AbstractModel.TensorRefLease qlnemb2 = preFFProjectionInput(lnpreFF)) {
                postFF = ffBlock.forward(qlnemb2.tensor(), tensorReducer, phase);
            }

            TensorRef lnpostFF = maybeApplyNorm(postFF, postFFNorm);

            applyResidual(lnpostFF, lnattn, "post_ff_residual");
            model.emitLayerDebug(layerIndex, "post_ff_residual", lnpostFF);

            if (lnemb != embedding) lnemb.close();
            if (lnpreFF != lnattn) {
                lnattn.close();
                lnpreFF.close();
            } else {
                lnattn.close();
            }

            return maybeApplyNorm(lnpostFF, preResponseNorm);
        }
    }

    private AbstractModel.TensorRefLease preAttentionProjectionInput(TensorRef tensor) {
        return model.maybeQuantizeReadOnly(tensor, "transformerblock.maybe_quantize.pre_attention");
    }

    private AbstractModel.TensorRefLease preFFProjectionInput(TensorRef tensor) {
        return model.maybeQuantizeReadOnly(tensor, "transformerblock.maybe_quantize.pre_ff");
    }

    private TensorRef maybeApplyNorm(TensorRef tensor, Optional<LayerNorm2> norm) {
        return norm.map(ln -> {
            TensorRef output = ln.forward(tensor);
            tensor.close();
            return output;
        }).orElse(tensor);
    }

    private void applyResidual(TensorRef target, TensorRef residual, String name) {
        compositeOps.addResidual(target, residual, model.getConfig().residualMultiplier == null
                ? 1.0f : model.getConfig().residualMultiplier);
    }

    private void emit(String stage, TensorRef tensor) {
        model.emitLayerDebug(layerIndex, stage, tensor);
    }
}
