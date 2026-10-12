package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Objects;

/** Executes the bias-free GELU gated MLP used by EmbeddingGemma 2. */
public final class EmbeddingGemma2MlpOperation {
    private final int hiddenSize;
    private final int intermediateSize;

    /** Creates a gated MLP operation for the configured hidden widths. */
    public EmbeddingGemma2MlpOperation(int hiddenSize, int intermediateSize) {
        if (hiddenSize <= 0 || intermediateSize <= 0) {
            throw new IllegalArgumentException("MLP dimensions must be positive");
        }
        this.hiddenSize = hiddenSize;
        this.intermediateSize = intermediateSize;
    }

    /** Computes {@code down(gelu(gate(input)) * up(input))} for every input row. */
    public void apply(TensorRef output, TensorRef input, TensorRef gateWeight, TensorRef upWeight,
                      TensorRef downWeight, Lighter lighter) {
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(gateWeight, "gateWeight");
        Objects.requireNonNull(upWeight, "upWeight");
        Objects.requireNonNull(downWeight, "downWeight");
        Objects.requireNonNull(lighter, "lighter");
        if (input.dims() != 2 || output.dims() != 2 || output.shape().first() != input.shape().first()
                || input.shape().last() != hiddenSize || output.shape().last() != hiddenSize
                || gateWeight.shape().first() != intermediateSize || gateWeight.shape().last() != hiddenSize
                || upWeight.shape().first() != intermediateSize || upWeight.shape().last() != hiddenSize
                || downWeight.shape().first() != hiddenSize || downWeight.shape().last() != intermediateSize) {
            throw new IllegalArgumentException("Invalid EmbeddingGemma 2 MLP tensor shapes");
        }
        try (TensorRef gate = lighter.allocate(DType.F32, TensorShape.of(input.shape().first(), intermediateSize));
             TensorRef up = lighter.allocate(DType.F32, TensorShape.of(input.shape().first(), intermediateSize))) {
            lighter.dotProductRows(gate, input, gateWeight, 0, hiddenSize, 0, intermediateSize, 0);
            lighter.dotProductRows(up, input, upWeight, 0, hiddenSize, 0, intermediateSize, 0);
            for (int row = 0; row < gate.shape().first(); row++) {
                for (int column = 0; column < intermediateSize; column++) {
                    float activated = ActivationFunction.eval(ActivationFunction.Type.GELU_PYTORCH_TANH,
                            gate.get(row, column));
                    gate.set(activated * up.get(row, column), row, column);
                }
            }
            lighter.dotProductRows(output, gate, downWeight, 0, intermediateSize, 0, hiddenSize, 0);
        }
    }
}
