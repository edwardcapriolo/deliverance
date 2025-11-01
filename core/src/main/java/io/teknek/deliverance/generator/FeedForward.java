package io.teknek.deliverance.generator;

import io.teknek.deliverance.tensor.AbstractTensor;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

public interface FeedForward {
    AbstractTensor forward(AbstractTensor input, Optional<Consumer<List<AbstractTensor>>> tensorReducer);
}