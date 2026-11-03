package io.teknek.reckoning.model.gemma4;

import io.teknek.reckoning.tensor.AbstractTensor;
import io.teknek.reckoning.tensor2.CompositeOps;
import io.teknek.reckoning.tensor2.MultiplyInPlace;
import io.teknek.reckoning.tensor2.TensorRef;

import java.util.function.BiFunction;

final class Gemma4PleSupport {
    private Gemma4PleSupport() {
    }

    /**
     * Upstream combines token-identity and projected PLE components as `(identity + projected) / sqrt(2)`.
     */
    static void combinePerLayerInputs(
            CompositeOps compositeOps,
            AbstractTensor projected,
            AbstractTensor tokenIdentity,
            float perLayerInputScale,
            int packedLength
    ) {
        try (TensorRef projectedRef = TensorRef.borrowed(projected);
             TensorRef identityRef = TensorRef.borrowed(tokenIdentity)) {
            compositeOps.addResidual(projectedRef, identityRef, 1.0f);
            compositeOps.multiplyInPlace(new MultiplyInPlace(perLayerInputScale).target(projectedRef)
                    .offsetAndLength(0, packedLength));
        }
    }

    static AbstractTensor[] splitPerLayerInputs(
            AbstractTensor projected,
            int batchSize,
            int numberOfLayers,
            int hiddenSizePerLayerInput,
            BiFunction<Integer, Integer, AbstractTensor> tensorFactory
    ) {
        AbstractTensor[] split = new AbstractTensor[numberOfLayers];
        for (int layer = 0; layer < numberOfLayers; layer++) {
            split[layer] = tensorFactory.apply(batchSize, hiddenSizePerLayerInput);
            int offset = layer * hiddenSizePerLayerInput;
            for (int b = 0; b < batchSize; b++) {
                split[layer].copyFrom(projected, projected.getOffset(b, offset), split[layer].getOffset(b, 0),
                        hiddenSizePerLayerInput);
            }
        }
        return split;
    }
}
