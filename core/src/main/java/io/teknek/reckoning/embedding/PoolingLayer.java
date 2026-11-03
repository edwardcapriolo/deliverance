package io.teknek.reckoning.embedding;

import io.teknek.reckoning.tensor.AbstractTensor;

import java.util.Optional;

public interface PoolingLayer {
    AbstractTensor getPoolingWeights();

    Optional<AbstractTensor> getPoolingBias();
}

