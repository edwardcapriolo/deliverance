package io.teknek.deliverance.generator2;

import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** TensorRef counterpart of {@code FeedForward}. */
public interface FeedForward2 {
    TensorRef forward(TensorRef input, Optional<Consumer<List<TensorRef>>> tensorReducer);

    default TensorRef forward(TensorRef input, Optional<Consumer<List<TensorRef>>> tensorReducer,
            ForwardPhase phase) {
        return forward(input, tensorReducer);
    }
}
