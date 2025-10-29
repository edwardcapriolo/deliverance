package io.teknek.deliverance.generator;


import io.teknek.deliverance.tensor.AbstractTensor;

public interface SampleOutput {

    LayerNorm getOutputLayerNorm();

    AbstractTensor getOutputLogitsWeights();
}