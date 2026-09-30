package io.teknek.deliverance.generator2;

import io.teknek.deliverance.tensor2.TensorRef;

/** Output-head weights owned by the TensorRef generation path. */
public interface SampleOutputRef extends AutoCloseable {
    LayerNorm2 outputLayerNorm();

    TensorRef outputLogitsWeights();

    @Override
    default void close() {
    }
}
