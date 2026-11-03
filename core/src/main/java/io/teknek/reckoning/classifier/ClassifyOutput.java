package io.teknek.reckoning.classifier;

import io.teknek.reckoning.tensor2.TensorRef;

import java.util.Optional;

public interface ClassifyOutput {
    TensorRef getClassificationWeights();

    Optional<TensorRef> getClassificationBias();

}
