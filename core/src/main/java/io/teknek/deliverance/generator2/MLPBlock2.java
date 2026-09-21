package io.teknek.deliverance.generator2;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.MultiplyAccumulate;
import io.teknek.deliverance.tensor2.TensorRef;

/** TensorRef-native SwiGLU MLP block used by Qwen3. */
public final class MLPBlock2 {
    private final AbstractModel model;
    private final TensorRef gateWeights;
    private final TensorRef upWeights;
    private final TensorRef downWeights;
    private final int hiddenLength;
    private final int embeddingLength;
    private final Lighter lighter;
    private final CompositeOps compositeOps;

    public MLPBlock2(AbstractModel model, TensorRef gateWeights, TensorRef upWeights, TensorRef downWeights,
            Lighter lighter) {
        this.model = java.util.Objects.requireNonNull(model, "model");
        this.gateWeights = java.util.Objects.requireNonNull(gateWeights, "gateWeights");
        this.upWeights = java.util.Objects.requireNonNull(upWeights, "upWeights");
        this.downWeights = java.util.Objects.requireNonNull(downWeights, "downWeights");
        this.hiddenLength = model.getConfig().hiddenLength;
        this.embeddingLength = model.getConfig().embeddingLength;
        this.lighter = java.util.Objects.requireNonNull(lighter, "lighter");
        this.compositeOps = new CompositeOps(lighter, model.getMetricRegistry());
    }

    public TensorRef forward(TensorRef input) {
        int batchSize = input.shape().first();
        TensorRef gate = model.makeDenseTensorRef(batchSize, hiddenLength);
        TensorRef up = model.makeDenseTensorRef(batchSize, hiddenLength);
        TensorRef output = model.makeDenseTensorRef(batchSize, embeddingLength);
        try {
            lighter.dotProductRows(gate, input, gateWeights, 0, embeddingLength, 0, hiddenLength, 0);
            lighter.dotProductRows(up, input, upWeights, 0, embeddingLength, 0, hiddenLength, 0);
            compositeOps.silu(gate);
            lighter.multiplyAccumulate(new MultiplyAccumulate(up).into(gate).offsetAndLength(0, hiddenLength));
            lighter.dotProductRows(output, gate, downWeights, 0, hiddenLength, 0, embeddingLength, 0);
            return output;
        } catch (RuntimeException | Error e) {
            output.close();
            throw e;
        } finally {
            gate.close();
            up.close();
        }
    }
}
