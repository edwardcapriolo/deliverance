package io.teknek.deliverance.generator;

import com.codahale.metrics.Timer;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensorlib.TensorPlan;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** HF BERT encoder block: residual is added before each post-attention/post-FF LayerNorm. */
public class BertTransformerBlock extends TransformerBlock {
    private final AbstractModel model;
    private final ConfigurableTensorProvider configurableTensorProvider;

    public BertTransformerBlock(AbstractModel model, int layerIndex, SelfAttention attention,
            LayerNorm postAttentionNorm, FeedForward ffBlock, LayerNorm postFFNorm,
            ConfigurableTensorProvider configurableTensorProvider) {
        super(model, layerIndex, Optional.empty(), attention, Optional.of(postAttentionNorm), Optional.empty(),
                ffBlock, Optional.of(postFFNorm), Optional.empty(), configurableTensorProvider);
        this.model = model;
        this.configurableTensorProvider = configurableTensorProvider;
    }

    @Override
    public AbstractTensor forward(AbstractTensor embedding, int position, KvBufferCache.KvBuffer kvBuffer,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer, ForwardPhase phase) {
        return forward(embedding, position, kvBuffer, tensorReducer, phase, 1, (int) embedding.shape().first(), null);
    }

    public AbstractTensor forward(AbstractTensor embedding, int position, KvBufferCache.KvBuffer kvBuffer,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer, ForwardPhase phase, int batchSize,
            int sequenceLength, int[] attentionMask) {
        Timer timer = InferenceProfiler.timer(model.getMetricRegistry(), "berttransformerblock.forward");
        try (Timer.Context ignored = timer.time()) {
            AbstractTensor postAttention;
            try (AbstractTensor qInput = model.maybeQuantizeReadOnly(embedding,
                    "berttransformerblock.maybe_quantize.attention")) {
                if (attention instanceof BertSelfAttention bertAttention) {
                    postAttention = bertAttention.forward(qInput, batchSize, sequenceLength, attentionMask,
                            tensorReducer, phase);
                } else {
                    postAttention = attention.forward(qInput, position, kvBuffer, tensorReducer, phase);
                }
            }
            addResidual(postAttention, embedding, "bert_post_attention_residual");
            AbstractTensor attentionOutput = postAttentionNorm.get().forward(postAttention);
            postAttention.close();
            model.emitLayerDebug(layerIndex, "post_attention_residual_norm", attentionOutput);

            AbstractTensor postFF;
            try (AbstractTensor qFfInput = model.maybeQuantizeReadOnly(attentionOutput,
                    "berttransformerblock.maybe_quantize.ff")) {
                postFF = ffBlock.forward(qFfInput, tensorReducer, phase);
            }
            addResidual(postFF, attentionOutput, "bert_post_ff_residual");
            AbstractTensor output = postFFNorm.get().forward(postFF);
            postFF.close();
            attentionOutput.close();
            model.emitLayerDebug(layerIndex, "post_ff_residual_norm", output);
            return output;
        }
    }

    private void addResidual(AbstractTensor target, AbstractTensor residual, String name) {
        TensorPlan plan = TensorPlanSupport.plan(model, configurableTensorProvider.get());
        plan.fuse(name, target.shape())
                .write("target", plan.mutable("target", target))
                .read("residual", plan.input("residual", residual))
                .map(name + " = residual(target, residual)", TensorPlan.TensorOp.CUSTOM,
                        (ctx, offset, length) -> TransformerBlock.applyResidualRange(ctx.tensor("target"),
                                ctx.tensor("residual"), model.getConfig().residualMultiplier, offset, length))
                .tensor()
                .materialize();
    }
}
