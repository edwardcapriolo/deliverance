package io.teknek.deliverance.guided;

import io.teknek.deliverance.model.ResponseContext;
import io.teknek.deliverance.tensor2.TensorRef;

/** TensorRef-native counterpart of {@link LogitsProcessor}. */
public interface LogitsProcessorRef {
    void process(TensorRef logits, ResponseContext responseContext);

    default void accept(int tokenId, ResponseContext responseContext) {
    }
}
