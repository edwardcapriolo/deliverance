package io.teknek.deliverance.model.nemotronlabsdiffusion;

import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.deliverance.generator2.KvCacheSelfAttention2;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

/** Nemotron V2 attention: generic GQA/cache mechanics with Nemotron YaRN RoPE. */
final class NemotronLabsDiffusionKvCacheSelfAttention2 extends KvCacheSelfAttention2 {
    private final NemotronLabsDiffusionRope rope;

    NemotronLabsDiffusionKvCacheSelfAttention2(AbstractModel model, int layerIndex,
            TensorRef queryWeights, TensorRef keyWeights, TensorRef valueWeights, TensorRef outputWeights,
            Lighter lighter, MetricRegistry metricRegistry, String queryName, String keyName, String valueName,
            String outputName, NemotronLabsDiffusionRope rope) {
        super(model, layerIndex, queryWeights, keyWeights, valueWeights, outputWeights, lighter, metricRegistry,
                queryName, keyName, valueName, outputName);
        this.rope = rope;
    }

    @Override
    protected void applyRotaryEmbedding(TensorRef query, TensorRef key, int startPosition) {
        rope.apply(query, key, startPosition, numberOfHeads, numberOfKeyValueHeads);
    }
}
