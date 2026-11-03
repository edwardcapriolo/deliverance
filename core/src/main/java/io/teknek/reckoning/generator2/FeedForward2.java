package io.teknek.reckoning.generator2;

import io.teknek.reckoning.generator.ForwardPhase;
import io.teknek.reckoning.tensor2.TensorRef;

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
