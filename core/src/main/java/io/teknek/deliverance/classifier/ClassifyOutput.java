package io.teknek.deliverance.classifier;

import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Optional;

public interface ClassifyOutput {
    TensorRef getClassificationWeights();

    Optional<TensorRef> getClassificationBias();

}
