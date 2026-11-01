package io.teknek.deliverance.model.diffusiongemma;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.Accumulate;
import io.teknek.deliverance.tensor2.ActivationMultiplyQuantize;
import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Objects;

/**
 * DiffusionGemma self-conditioning block, composed from TensorPlan/tensor primitives.
 *
 * <p>Matches the HF data flow:</p>
 *
 * <pre>{@code
 * normed = pre_norm(self_conditioning_signal)
 * sc_signal = down_proj(act(gate_proj(normed)) * up_proj(normed))
 * combined = inputs_embeds + sc_signal
 * output = post_norm(combined)
 * }</pre>
 *
 * <p>Current gap: flattening between 3D canvas tensors and 2D projection tensors still copies data. TensorPlan can later
 * fuse these layout transitions; this class keeps the math expressed through existing tensor operations rather than
 * adding model-specific provider methods.</p>
 */
public final class DiffusionGemmaSelfConditioning {
    private final DiffusionGemmaTextConfig config;
    private final TensorRef preNormWeightRef;
    private final TensorRef gateWeightRef;
    private final TensorRef upWeightRef;
    private final TensorRef downWeightRef;
    private final Lighter lighter;
    private final CompositeOps compositeOps;

    /** TensorRef/Lighter implementation of the self-conditioning block. */
    public DiffusionGemmaSelfConditioning(DiffusionGemmaTextConfig config, TensorRef preNormWeight,
            TensorRef gateWeight, TensorRef upWeight, TensorRef downWeight, Lighter lighter,
            CompositeOps compositeOps) {
        this.config = Objects.requireNonNull(config, "config");
        this.preNormWeightRef = Objects.requireNonNull(preNormWeight, "preNormWeight");
        this.lighter = Objects.requireNonNull(lighter, "lighter");
        this.compositeOps = Objects.requireNonNull(compositeOps, "compositeOps");
        this.gateWeightRef = gateWeight;
        this.upWeightRef = upWeight;
        this.downWeightRef = downWeight;
    }

    public TensorRef forward(TensorRef inputsEmbeds, TensorRef selfConditioningSignal) {
        validateCanvasEmbeddings(inputsEmbeds, "inputsEmbeds");
        validateCanvasEmbeddings(selfConditioningSignal, "selfConditioningSignal");
        Preconditions.checkArgument(inputsEmbeds.shape().equals(selfConditioningSignal.shape()),
                "inputsEmbeds and selfConditioningSignal shapes must match");
        int batch = inputsEmbeds.shape().dim(0);
        int canvas = inputsEmbeds.shape().dim(1);
        int hidden = config.hiddenSize;
        int rows = batch * canvas;
        try (TensorRef normed = lighter.allocate(DType.F32, inputsEmbeds.shape());
             TensorRef normed2d = lighter.allocate(DType.F32, TensorShape.of(rows, hidden));
             TensorRef gate = lighter.allocate(DType.F32, TensorShape.of(rows, config.intermediateSize));
             TensorRef up = lighter.allocate(DType.F32, TensorShape.of(rows, config.intermediateSize));
             TensorRef down = lighter.allocate(DType.F32, TensorShape.of(rows, hidden));
             TensorRef combined = lighter.allocate(DType.F32, TensorShape.of(rows, hidden))) {
            rmsNorm3d(normed, selfConditioningSignal, preNormWeightRef);
            flatten(normed, normed2d);
            project(normed2d, gate, gateWeightRef, config.intermediateSize, hidden);
            project(normed2d, up, upWeightRef, config.intermediateSize, hidden);
            try (TensorRef activated = compositeOps.activationMultiplyQuantize(new ActivationMultiplyQuantize(
                    gate, up, config.hiddenActivation, DType.F32).offsetAndLength(0, config.intermediateSize))) {
                project(activated, down, downWeightRef, hidden, config.intermediateSize);
            }
            flatten(inputsEmbeds, combined);
            lighter.accumulate(new Accumulate(down).into(combined).offsetAndLength(0, hidden));
            TensorRef output = lighter.allocate(DType.F32, inputsEmbeds.shape());
            try {
                inflate(combined, output, batch, canvas, hidden);
                rmsNorm3d(output, output, null);
                return output;
            } catch (RuntimeException | Error error) {
                output.close();
                throw error;
            }
        }
    }

    private void project(TensorRef input, TensorRef output, TensorRef weight, int outputLength, int inputLength) {
        lighter.dotProductRows(output, input, weight, 0, inputLength, 0, outputLength, 0);
    }

    private void rmsNorm(TensorRef output, TensorRef input, TensorRef weight) {
        int hidden = config.hiddenSize;
        for (int row = 0; row < input.shape().first(); row++) {
            double sum = 0.0;
            for (int column = 0; column < hidden; column++) {
                float value = input.get(row, column);
                sum += value * value;
            }
            float scale = (float) (1.0 / StrictMath.sqrt(sum / hidden + config.rmsNormEps));
            for (int column = 0; column < hidden; column++) {
                output.set(input.get(row, column) * scale * (weight == null ? 1.0f : weight.get(0, column)), row,
                        column);
            }
        }
    }

    private void rmsNorm3d(TensorRef output, TensorRef input, TensorRef weight) {
        int hidden = config.hiddenSize;
        for (int batch = 0; batch < input.shape().dim(0); batch++) {
            for (int position = 0; position < input.shape().dim(1); position++) {
                double sum = 0.0;
                for (int column = 0; column < hidden; column++) {
                    float value = input.get(batch, position, column);
                    sum += value * value;
                }
                float scale = (float) (1.0 / StrictMath.sqrt(sum / hidden + config.rmsNormEps));
                for (int column = 0; column < hidden; column++) {
                    output.set(input.get(batch, position, column) * scale
                            * (weight == null ? 1.0f : weight.get(0, column)), batch, position, column);
                }
            }
        }
    }

    private void flatten(TensorRef source, TensorRef target) {
        int row = 0;
        for (int batch = 0; batch < source.shape().dim(0); batch++) {
            for (int position = 0; position < source.shape().dim(1); position++) {
                for (int column = 0; column < source.shape().dim(2); column++) {
                    target.set(source.get(batch, position, column), row, column);
                }
                row++;
            }
        }
    }

    private void inflate(TensorRef source, TensorRef target, int batch, int canvas, int hidden) {
        int row = 0;
        for (int b = 0; b < batch; b++) {
            for (int position = 0; position < canvas; position++) {
                for (int column = 0; column < hidden; column++) {
                    target.set(source.get(row, column), b, position, column);
                }
                row++;
            }
        }
    }



    private void validateCanvasEmbeddings(TensorRef tensor, String name) {
        Preconditions.checkArgument(tensor.dims() == 3, name + " must be [batch, canvas, hidden]");
        Preconditions.checkArgument(tensor.shape().dim(2) == config.hiddenSize,
                name + " hidden dimension must match config.hiddenSize");
    }

}
