package io.teknek.reckoning.guided;

import io.teknek.reckoning.model.ResponseContext;
import io.teknek.reckoning.tensor2.TensorRef;

/** TensorRef-native counterpart of {@link LogitsProcessor}. */
public interface LogitsProcessorRef {
    void process(TensorRef logits, ResponseContext responseContext);

    default void accept(int tokenId, ResponseContext responseContext) {
    }
}
