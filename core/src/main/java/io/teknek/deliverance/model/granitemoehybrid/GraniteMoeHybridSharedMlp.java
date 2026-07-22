package io.teknek.deliverance.model.granitemoehybrid;

import com.codahale.metrics.Timer;
import io.teknek.deliverance.generator.FeedForward;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

public class GraniteMoeHybridSharedMlp implements FeedForward {

    private final AbstractModel model;
    private final GraniteMoeHybridConfig config;
    private final AbstractTensor inputLinearWeights;
    private final AbstractTensor outputLinearWeights;
    private final ConfigurableTensorProvider configurableTensorProvider;

    public GraniteMoeHybridSharedMlp(AbstractModel model, GraniteMoeHybridConfig config,
            AbstractTensor inputLinearWeights, AbstractTensor outputLinearWeights,
            ConfigurableTensorProvider configurableTensorProvider) {
        this.model = model;
        this.config = config;
        this.inputLinearWeights = inputLinearWeights;
        this.outputLinearWeights = outputLinearWeights;
        this.configurableTensorProvider = configurableTensorProvider;
        configurableTensorProvider.get().registerModelTensor(inputLinearWeights);
        configurableTensorProvider.get().registerModelTensor(outputLinearWeights);
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        try (Timer.Context ignored = InferenceProfiler.timer(this.model.getMetricRegistry(),
                "granitemoehybrid.shared_mlp.forward").time()) {
            int batchSize = input.shape().first();
            try (AbstractTensor projected = this.model.getTensorAllocator()
                    .getDirty(this.model.getWorkingDType(), TensorShape.of(batchSize, this.config.sharedIntermediateSize * 2));
                 AbstractTensor hidden = this.model.getTensorAllocator()
                         .getDirty(this.model.getWorkingDType(), TensorShape.of(batchSize, this.config.sharedIntermediateSize))) {
                VectorMath.pchunk(0, this.config.sharedIntermediateSize * 2, (chunkStart, chunkSize) ->
                        this.configurableTensorProvider.get().dotProductChunk(projected, input, this.inputLinearWeights,
                                0, this.config.embeddingLength, chunkStart, chunkSize),
                        this.configurableTensorProvider.get().parallelSplitSize(), this.model.getPool());

                applyActivationGate(projected, hidden, this.config.activationFunction, this.config.sharedIntermediateSize);
                tensorReducer.ifPresent(func -> func.accept(List.of(hidden)));

                try (AbstractTensor hiddenQ = this.model.maybeQuantize(hidden)) {
                    AbstractTensor output = this.model.makeTensor(batchSize, this.config.embeddingLength);
                    VectorMath.pchunk(0, this.config.embeddingLength, (chunkStart, chunkSize) ->
                            this.configurableTensorProvider.get().dotProductChunk(output, hiddenQ, this.outputLinearWeights,
                                    0, this.config.sharedIntermediateSize, chunkStart, chunkSize),
                            this.configurableTensorProvider.get().parallelSplitSize(), this.model.getPool());
                    return output;
                }
            }
        }
    }

    /**
     * Applies the GraniteMoeHybrid shared MLP gate after {@code input_linear}.
     *
     * <p>Transformers computes this as:</p>
     * <pre>
     * hidden_states = input_linear(hidden_states)
     * gate, value = hidden_states.chunk(2, dim=-1)
     * hidden_states = silu(gate) * value
     * </pre>
     *
     * <p>This helper is intentionally local to the feed-forward implementation rather than
     * exposed as a {@code TensorOperations} provider contract. The F32 path uses the Java
     * Vector API because this operation is on the per-token hot path. If we later add a
     * reusable provider-level fused GLU primitive, this is the call site to replace.</p>
     */
    static void applyActivationGate(AbstractTensor projected, AbstractTensor hidden,
            ActivationFunction.Type activationFunction, int sharedIntermediateSize) {
        if (activationFunction == ActivationFunction.Type.SILU && projected instanceof FloatBufferTensor projectedF32
                && hidden instanceof FloatBufferTensor hiddenF32) {
            applySiluGateF32(projectedF32, hiddenF32, sharedIntermediateSize);
            return;
        }
        applyActivationGateScalar(projected, hidden, activationFunction, sharedIntermediateSize);
    }

    private static void applySiluGateF32(FloatBufferTensor projected, FloatBufferTensor hidden,
            int sharedIntermediateSize) {
        int batchSize = projected.shape().first();
        int upperBound = FloatVector.SPECIES_PREFERRED.loopBound(sharedIntermediateSize);
        FloatVector one = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, 1.0f);
        for (int row = 0; row < batchSize; row++) {
            int col = 0;
            for (; col < upperBound; col += FloatVector.SPECIES_PREFERRED.length()) {
                FloatVector gate = projected.getVector(FloatVector.SPECIES_PREFERRED, row, col);
                FloatVector value = projected.getVector(FloatVector.SPECIES_PREFERRED, row, col + sharedIntermediateSize);
                FloatVector sigmoid = one.div(gate.neg().lanewise(VectorOperators.EXP).add(one));
                hidden.intoTensor(gate.mul(sigmoid).mul(value), row, col);
            }
            for (; col < sharedIntermediateSize; col++) {
                float gate = projected.get(row, col);
                float value = projected.get(row, col + sharedIntermediateSize);
                hidden.set(ActivationFunction.eval(ActivationFunction.Type.SILU, gate) * value, row, col);
            }
        }
    }

    private static void applyActivationGateScalar(AbstractTensor projected, AbstractTensor hidden,
            ActivationFunction.Type activationFunction, int sharedIntermediateSize) {
        int batchSize = projected.shape().first();
        for (int row = 0; row < batchSize; row++) {
            for (int col = 0; col < sharedIntermediateSize; col++) {
                float gate = projected.get(row, col);
                float value = projected.get(row, col + sharedIntermediateSize);
                hidden.set(ActivationFunction.eval(activationFunction, gate) * value, row, col);
            }
        }
    }

}
