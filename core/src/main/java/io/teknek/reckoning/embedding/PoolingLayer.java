package io.teknek.reckoning.embedding;

import io.teknek.reckoning.tensor2.TensorRef;

import java.util.Optional;

public interface PoolingLayer {
    TensorRef getPoolingWeights();

    Optional<TensorRef> getPoolingBias();
}
