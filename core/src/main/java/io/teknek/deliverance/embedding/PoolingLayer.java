package io.teknek.deliverance.embedding;

import io.teknek.deliverance.tensor.AbstractTensor;

import java.util.Optional;

public interface PoolingLayer {
    AbstractTensor getPoolingWeights();

    Optional<AbstractTensor> getPoolingBias();
}

