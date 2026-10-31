package io.teknek.deliverance.generator2;

import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.dropwizard.metrics5.MetricRegistry;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** GPT2 causal attention: KVCache2 attention with the checkpoint's projection biases. */
public final class Gpt2CausalSelfAttention2 extends KvCacheSelfAttention2 {
    private final TensorRef queryBias;
    private final TensorRef keyBias;
    private final TensorRef valueBias;
    private final TensorRef outputBias;

    public Gpt2CausalSelfAttention2(AbstractModel model, int layerIndex,
            TensorRef queryWeight, TensorRef keyWeight, TensorRef valueWeight, TensorRef outputWeight,
            TensorRef queryBias, TensorRef keyBias, TensorRef valueBias, TensorRef outputBias,
            Lighter lighter, MetricRegistry metricRegistry) {
        super(model, layerIndex, queryWeight, keyWeight, valueWeight, outputWeight, lighter, metricRegistry,
                null, null, null, null);
        this.queryBias = queryBias;
        this.keyBias = keyBias;
        this.valueBias = valueBias;
        this.outputBias = outputBias;
    }

    @Override
    protected TensorRef[] projectQkv(TensorRef input, ForwardPhase phase) {
        TensorRef[] qkv = super.projectQkv(input, phase);
        addBias(qkv[0], queryBias);
        addBias(qkv[1], keyBias);
        addBias(qkv[2], valueBias);
        return qkv;
    }

    @Override
    protected void applyOutputProjectionBias(TensorRef output) {
        addBias(output, outputBias);
    }

    private void addBias(TensorRef target, TensorRef bias) {
        for (int row = 0; row < target.shape().first(); row++) {
            for (int column = 0; column < target.shape().last(); column++) {
                target.set(target.get(row, column) + bias.get(0, column), row, column);
            }
        }
    }
}
