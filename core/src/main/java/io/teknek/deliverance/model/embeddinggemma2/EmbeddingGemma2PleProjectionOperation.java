package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor2.GemmDit;
import io.teknek.deliverance.tensorlib.TensorPlanAdaptiveSplitTuner;

import java.util.Objects;

/**
 * Projects scaled token embeddings into EmbeddingGemma 2 per-layer inputs.
 *
 * <p>The reference operation is a learned projection followed by
 * {@code hiddenSize^-0.5} scaling. The layer-axis reshape and RMSNorm are
 * deliberately separate operations so each PLE contract can be proven in
 * isolation.</p>
 */
public final class EmbeddingGemma2PleProjectionOperation {
    private final int inputWidth;
    private final int outputWidth;
    private final float scale;

    /**
     * Creates a PLE projection operation.
     *
     * @param inputWidth token embedding width
     * @param numberOfLayers number of per-layer PLE slices
     * @param perLayerInputWidth width of each PLE slice
     */
    public EmbeddingGemma2PleProjectionOperation(int inputWidth, int numberOfLayers, int perLayerInputWidth) {
        if (inputWidth <= 0 || numberOfLayers <= 0 || perLayerInputWidth <= 0) {
            throw new IllegalArgumentException("PLE dimensions must be positive");
        }
        this.inputWidth = inputWidth;
        this.outputWidth = Math.multiplyExact(numberOfLayers, perLayerInputWidth);
        this.scale = (float) (1.0 / Math.sqrt(inputWidth));
    }

    /**
     * Projects rows shaped {@code [tokens, inputWidth]} using weights shaped
     * {@code [outputWidth, inputWidth]} into {@code [tokens, outputWidth]}.
     *
     * @param output writable projected output
     * @param input input token embeddings
     * @param projection learned PLE projection matrix
     * @param lighter TensorRef operation/provider owner
     */
    public void apply(TensorRef output, TensorRef input, TensorRef projection, Lighter lighter) {
        apply(output, input, projection, lighter, null);
    }

    public void apply(TensorRef output, TensorRef input, TensorRef projection, Lighter lighter,
            TensorPlanAdaptiveSplitTuner tuner) {
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(projection, "projection");
        Objects.requireNonNull(lighter, "lighter");
        if (input.dims() != 2 || projection.dims() != 2 || output.dims() != 2
                || input.shape().last() != inputWidth
                || projection.shape().first() != outputWidth
                || projection.shape().last() != inputWidth
                || output.shape().first() != input.shape().first()
                || output.shape().last() != outputWidth) {
            throw new IllegalArgumentException("Invalid EmbeddingGemma 2 PLE projection shapes");
        }
        GemmDit.dotProductRows(lighter, tuner, "embeddinggemma2.ple_projection", output, input, projection,
                0, 0, inputWidth, 0, outputWidth, 0);
        for (int row = 0; row < output.shape().first(); row++) {
            try (TensorRef outputRow = output.slice(row)) {
                lighter.preferredTensorOperations().scale(scale, outputRow, 0, outputWidth);
            }
        }
    }
}
