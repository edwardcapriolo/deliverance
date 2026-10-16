package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor2.GemmDit;
import io.teknek.deliverance.tensorlib.TensorPlanAdaptiveSplitTuner;

import java.util.Objects;

/** Executes the bias-free GELU gated MLP used by EmbeddingGemma 2. */
public final class EmbeddingGemma2MlpOperation {
    private final int hiddenSize;
    private final int intermediateSize;
    private final boolean roundBf16;

    /** Creates a gated MLP operation for the configured hidden widths. */
    public EmbeddingGemma2MlpOperation(int hiddenSize, int intermediateSize) {
        this(hiddenSize, intermediateSize, false);
    }

    /** Creates a MLP with optional BF16 activation-boundary rounding matching the reference checkpoint. */
    public EmbeddingGemma2MlpOperation(int hiddenSize, int intermediateSize, boolean roundBf16) {
        if (hiddenSize <= 0 || intermediateSize <= 0) {
            throw new IllegalArgumentException("MLP dimensions must be positive");
        }
        this.hiddenSize = hiddenSize;
        this.intermediateSize = intermediateSize;
        this.roundBf16 = roundBf16;
    }

    /** Computes {@code down(gelu(gate(input)) * up(input))} for every input row. */
    public void apply(TensorRef output, TensorRef input, TensorRef gateWeight, TensorRef upWeight,
                      TensorRef downWeight, Lighter lighter) {
        apply(output, input, gateWeight, upWeight, downWeight, lighter, null);
    }

    public void apply(TensorRef output, TensorRef input, TensorRef gateWeight, TensorRef upWeight,
                      TensorRef downWeight, Lighter lighter, TensorPlanAdaptiveSplitTuner tuner) {
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
            GemmDit.dotProductRows(lighter, tuner, "embeddinggemma2.mlp.gate", gate, input, gateWeight,
                    0, 0, hiddenSize, 0, intermediateSize, 0);
            GemmDit.dotProductRows(lighter, tuner, "embeddinggemma2.mlp.up", up, input, upWeight,
                    0, 0, hiddenSize, 0, intermediateSize, 0);
            roundBoundary(gate, lighter);
            roundBoundary(up, lighter);
            for (int row = 0; row < gate.shape().first(); row++) {
                for (int column = 0; column < intermediateSize; column++) {
                    float activated = ActivationFunction.eval(ActivationFunction.Type.GELU_PYTORCH_TANH,
                            gate.get(row, column));
                    gate.set(activated * up.get(row, column), row, column);
                }
            }
            roundBoundary(gate, lighter);
            GemmDit.dotProductRows(lighter, tuner, "embeddinggemma2.mlp.down", output, gate, downWeight,
                    0, 0, intermediateSize, 0, hiddenSize, 0);
            roundBoundary(output, lighter);
        }
    }

    private void roundBoundary(TensorRef target, Lighter lighter) {
        if (!roundBf16) return;
        try (TensorRef bf16 = lighter.reshape(target, DType.BF16,
                io.teknek.deliverance.tensor2.Threading.PROVIDER)) {
            lighter.reshape(bf16, target, io.teknek.deliverance.tensor2.Threading.PROVIDER);
        }
    }
}
