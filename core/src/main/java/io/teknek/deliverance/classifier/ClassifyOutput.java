package io.teknek.deliverance.classifier;

import io.teknek.deliverance.tensor.AbstractTensor;

import java.util.Optional;

public interface ClassifyOutput {
    AbstractTensor getClassificationWeights();

    Optional<AbstractTensor> getClassificationBias();

}
