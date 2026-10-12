package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.MultiplyAccumulate;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Objects;

/** Executes one batched EmbeddingGemma 2 PLE residual block. */
public final class EmbeddingGemma2PleBlockOperation {
    private final int hiddenSize;
    private final int perLayerInputWidth;
    private final float epsilon;
    private final EmbeddingGemma2RmsNormOperation postNorm;

    /**
     * Creates a PLE residual block operation.
     *
     * @param hiddenSize hidden width of the text backbone
     * @param epsilon RMSNorm epsilon
     */
    public EmbeddingGemma2PleBlockOperation(int hiddenSize, float epsilon) {
        this(hiddenSize, hiddenSize, epsilon);
    }

    /** Creates a PLE block where the per-layer signal may be narrower than the residual stream. */
    public EmbeddingGemma2PleBlockOperation(int hiddenSize, int perLayerInputWidth, float epsilon) {
        if (hiddenSize <= 0 || !Float.isFinite(epsilon) || epsilon < 0.0f) {
            throw new IllegalArgumentException("Invalid PLE block configuration");
        }
        if (perLayerInputWidth <= 0) {
            throw new IllegalArgumentException("PLE input width must be positive");
        }
        this.hiddenSize = hiddenSize;
        this.perLayerInputWidth = perLayerInputWidth;
        this.epsilon = epsilon;
        this.postNorm = new EmbeddingGemma2RmsNormOperation(epsilon);
    }

    /**
     * Applies the PLE block to all rows in one batch.
     *
     * @param output writable output shaped {@code [tokens, hiddenSize]}
     * @param hiddenStates input residual stream
     * @param perLayerInput normalized PLE input for this layer
     * @param gateWeight gate projection shaped {@code [hiddenSize, hiddenSize]}
     * @param projectionWeight output projection shaped {@code [hiddenSize, hiddenSize]}
     * @param normWeight post-PLE RMSNorm scale shaped {@code [1, hiddenSize]}
     * @param lighter TensorRef operation/provider owner
     */
    public void apply(TensorRef output, TensorRef hiddenStates, TensorRef perLayerInput,
                      TensorRef gateWeight, TensorRef projectionWeight, TensorRef normWeight,
                      Lighter lighter) {
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(hiddenStates, "hiddenStates");
        Objects.requireNonNull(perLayerInput, "perLayerInput");
        Objects.requireNonNull(gateWeight, "gateWeight");
        Objects.requireNonNull(projectionWeight, "projectionWeight");
        Objects.requireNonNull(normWeight, "normWeight");
        Objects.requireNonNull(lighter, "lighter");
        validateShapes(output, hiddenStates, perLayerInput, gateWeight, projectionWeight, normWeight);

        try (TensorRef residual = lighter.allocate(DType.F32, TensorShape.of(hiddenStates.shape().first(), hiddenSize));
             TensorRef gated = lighter.allocate(DType.F32,
                     TensorShape.of(hiddenStates.shape().first(), perLayerInputWidth))) {
            lighter.copyStorage(hiddenStates, residual);
            lighter.dotProductRows(gated, hiddenStates, gateWeight, 0, hiddenSize, 0, perLayerInputWidth, 0);
            applyGelu(gated);
            lighter.multiplyAccumulate(new MultiplyAccumulate(perLayerInput).into(gated)
                    .offsetAndLength(0, perLayerInputWidth));
            lighter.dotProductRows(output, gated, projectionWeight, 0, perLayerInputWidth, 0, hiddenSize, 0);
            postNorm.apply(output, output, normWeight, lighter);
            new CompositeOps(lighter).addResidual(output, residual, 1.0f);
        }
    }

    private void applyGelu(TensorRef target) {
        for (int row = 0; row < target.shape().first(); row++) {
            for (int column = 0; column < target.shape().last(); column++) {
                target.set(ActivationFunction.eval(ActivationFunction.Type.GELU_PYTORCH_TANH,
                        target.get(row, column)), row, column);
            }
        }
    }

    private void validateShapes(TensorRef output, TensorRef hiddenStates, TensorRef perLayerInput,
                                TensorRef gateWeight, TensorRef projectionWeight, TensorRef normWeight) {
        if (output.dims() != 2 || hiddenStates.dims() != 2 || perLayerInput.dims() != 2
                || output.shape().equals(hiddenStates.shape()) == false
                || perLayerInput.shape().first() != hiddenStates.shape().first()
                || perLayerInput.shape().last() != perLayerInputWidth
                || output.shape().last() != hiddenSize
                || gateWeight.dims() != 2 || projectionWeight.dims() != 2
                || gateWeight.shape().first() != perLayerInputWidth || gateWeight.shape().last() != hiddenSize
                || projectionWeight.shape().first() != hiddenSize || projectionWeight.shape().last() != perLayerInputWidth
                || normWeight.dims() != 2 || normWeight.shape().first() != 1
                || normWeight.shape().last() < hiddenSize) {
            throw new IllegalArgumentException("Invalid EmbeddingGemma 2 PLE block shapes");
        }
    }
}
