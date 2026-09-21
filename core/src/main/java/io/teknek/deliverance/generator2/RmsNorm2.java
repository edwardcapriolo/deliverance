package io.teknek.deliverance.generator2;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.generator.Gemma4RmsNormSupport;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor2.TensorRefBackedTensor;

/** TensorRef-native RMSNorm layer. */
public final class RmsNorm2 {
    private final AbstractModel model;
    private final TensorRef weights;
    private final float weightAdjustment;
    private final float epsilon;
    private final CompositeOps compositeOps;

    public RmsNorm2(AbstractModel model, TensorRef weights, float weightAdjustment) {
        this.model = java.util.Objects.requireNonNull(model, "model");
        this.weights = java.util.Objects.requireNonNull(weights, "weights");
        this.weightAdjustment = weightAdjustment;
        this.epsilon = model.getConfig().layerNormEps;
        this.compositeOps = new CompositeOps(model.getLighter(), model.getMetricRegistry());
    }

    public TensorRef forward(TensorRef input) {
        TensorRef output = model.makeDenseTensorRef(input.shape());
        try {
            if (input.dType() == io.teknek.deliverance.DType.F32
                    && output.dType() == io.teknek.deliverance.DType.F32
                    && weights.dType() == io.teknek.deliverance.DType.F32) {
                compositeOps.rmsNorm(output, input, weights, epsilon, weightAdjustment);
            } else {
                AbstractTensor inputTensor = new TensorRefBackedTensor(input);
                AbstractTensor outputTensor = new TensorRefBackedTensor(output);
                AbstractTensor weightTensor = new TensorRefBackedTensor(weights);
                // The helper operates in place, so copy the input into the output first for mixed dtypes.
                outputTensor.copyFrom(inputTensor, 0, 0, Math.toIntExact(input.shape().size()));
                Gemma4RmsNormSupport.applyInPlaceSimd(outputTensor, 1, input.shape().last(), epsilon, weightTensor);
            }
            return output;
        } catch (RuntimeException | Error e) {
            output.close();
            throw e;
        }
    }
}
