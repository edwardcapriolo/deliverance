package io.teknek.deliverance.generator;

import com.codahale.metrics.Timer;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

public class Gemma4TransformerBlock extends TransformerBlock {
    private final AbstractModel model;
    private final ConfigurableTensorProvider configurableTensorProvider;
    private final Optional<AbstractTensor> perLayerInputGateWeights;
    private final Optional<AbstractTensor> perLayerProjectionWeights;
    private final Optional<LayerNorm> postPerLayerInputNorm;
    private final int perLayerInputLength;
    private final float layerScalar;
    private final ActivationFunction.Type activationFunction;

    public Gemma4TransformerBlock(
            AbstractModel model,
            int layerIndex,
            LayerNorm preAttentionNorm,
            CausalSelfAttention attention,
            LayerNorm postAttentionNorm,
            LayerNorm preFFNorm,
            FeedForward ffBlock,
            LayerNorm postFFNorm,
            ConfigurableTensorProvider configurableTensorProvider,
            Optional<AbstractTensor> perLayerInputGateWeights,
            Optional<AbstractTensor> perLayerProjectionWeights,
            Optional<LayerNorm> postPerLayerInputNorm,
            int perLayerInputLength,
            float layerScalar,
            ActivationFunction.Type activationFunction
    ) {
        super(model, layerIndex, preAttentionNorm, attention, postAttentionNorm, preFFNorm, ffBlock, postFFNorm,
                configurableTensorProvider);
        this.model = model;
        this.configurableTensorProvider = configurableTensorProvider;
        this.perLayerInputGateWeights = perLayerInputGateWeights;
        this.perLayerProjectionWeights = perLayerProjectionWeights;
        this.postPerLayerInputNorm = postPerLayerInputNorm;
        this.perLayerInputLength = perLayerInputLength;
        this.layerScalar = layerScalar;
        this.activationFunction = activationFunction;
    }

    public AbstractTensor forward(
            AbstractTensor embedding,
            AbstractTensor perLayerInput,
            int position,
            KvBufferCache.KvBuffer kvBuffer,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer
    ) {
        Timer.Context totalTimer = model.getMetricRegistry().timer("gemma4.block." + layerIndex + ".total").time();
        try {
        AbstractTensor lnemb = preAttentionNorm.map(ln -> ln.forward(embedding)).orElse(embedding);
        AbstractTensor postAttention;
        try (AbstractTensor qlnemb = model.maybeQuantize(lnemb)) {
            postAttention = attention.forward(qlnemb, position, kvBuffer, tensorReducer);
        }
        AbstractTensor lnattn = maybeApplyNorm(postAttention, postAttentionNorm);
        if (model.getConfig().residualMultiplier != null) {
            configurableTensorProvider.get().scale(model.getConfig().residualMultiplier, lnattn, 0, model.getConfig().embeddingLength);
        }
        configurableTensorProvider.get().accumulate(lnattn, embedding, 0, model.getConfig().embeddingLength);

        AbstractTensor lnpreFF = preFFNorm.map(ln -> ln.forward(lnattn)).orElse(lnattn);
        AbstractTensor postFF;
        try (AbstractTensor qlnemb2 = model.maybeQuantize(lnpreFF)) {
            postFF = ffBlock.forward(qlnemb2, tensorReducer);
        }

        AbstractTensor lnpostFF = maybeApplyNorm(postFF, postFFNorm);
        if (model.getConfig().residualMultiplier != null) {
            configurableTensorProvider.get().scale(model.getConfig().residualMultiplier, lnpostFF, 0, model.getConfig().embeddingLength);
        }
        configurableTensorProvider.get().accumulate(lnpostFF, lnattn, 0, model.getConfig().embeddingLength);

        if (perLayerInput != null && perLayerInputGateWeights.isPresent() && perLayerProjectionWeights.isPresent() && postPerLayerInputNorm.isPresent()) {
            try (
                    AbstractTensor gated = model.makeDenseTensor(lnpostFF.shape().first(), perLayerInputLength);
                    AbstractTensor projected = model.makeDenseTensor(lnpostFF.shape().first(), model.getConfig().embeddingLength)
            ) {
                configurableTensorProvider.get().dotProductChunk(gated, lnpostFF, perLayerInputGateWeights.get(), 0,
                        model.getConfig().embeddingLength, 0, perLayerInputLength);
                for (int b = 0; b < gated.shape().first(); b++) {
                    for (int i = 0; i < perLayerInputLength; i++) {
                        float v = ActivationFunction.eval(activationFunction, gated.get(b, i)) * perLayerInput.get(b, i);
                        gated.set(v, b, i);
                    }
                }
                try (AbstractTensor gatedQ = model.maybeQuantize(gated)) {
                    configurableTensorProvider.get().dotProductChunk(projected, gatedQ, perLayerProjectionWeights.get(), 0,
                            perLayerInputLength, 0, model.getConfig().embeddingLength);
                }
                AbstractTensor normalized = postPerLayerInputNorm.get().forward(projected);
                configurableTensorProvider.get().accumulate(normalized, lnpostFF, 0, model.getConfig().embeddingLength);
                lnpostFF.close();
                lnpostFF = normalized;
            }
        }

        if (layerScalar != 1.0f) {
            configurableTensorProvider.get().scale(layerScalar, lnpostFF, 0, model.getConfig().embeddingLength);
        }

        if (lnemb != embedding) lnemb.close();
        if (lnattn != postAttention) lnattn.close();
        else postAttention.close();
        if (lnpreFF != lnattn) lnpreFF.close();
        else lnattn.close();

        return lnpostFF;
        } finally {
            totalTimer.stop();
        }
    }

    private AbstractTensor maybeApplyNorm(AbstractTensor tensor, Optional<LayerNorm> norm) {
        return norm.map(ln -> {
            AbstractTensor o = ln.forward(tensor);
            tensor.close();
            return o;
        }).orElse(tensor);
    }
}
