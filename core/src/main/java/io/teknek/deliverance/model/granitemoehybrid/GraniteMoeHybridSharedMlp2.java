package io.teknek.deliverance.model.granitemoehybrid;

import io.dropwizard.metrics5.Timer;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.generator2.FeedForward2;
import io.teknek.deliverance.tensor2.BatchDotProduct;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** TensorRef implementation of Granite's packed gated shared MLP. */
final class GraniteMoeHybridSharedMlp2 implements FeedForward2 {
    private final AbstractModel model;
    private final int layerIndex;
    private final TensorRef inputWeights;
    private final TensorRef outputWeights;
    private final int hiddenLength;
    private final int embeddingLength;

    GraniteMoeHybridSharedMlp2(AbstractModel model, TensorRef inputWeights, TensorRef outputWeights) {
        this(model, inputWeights, outputWeights, -1);
    }

    GraniteMoeHybridSharedMlp2(AbstractModel model, TensorRef inputWeights, TensorRef outputWeights,
            int layerIndex) {
        this.model = model;
        this.layerIndex = layerIndex;
        this.inputWeights = inputWeights;
        this.outputWeights = outputWeights;
        this.hiddenLength = ((GraniteMoeHybridConfig) model.getConfig()).sharedIntermediateSize;
        this.embeddingLength = model.getConfig().embeddingLength;
    }

    @Override
    public TensorRef forward(TensorRef input, Optional<Consumer<List<TensorRef>>> tensorReducer) {
        return forward(input, tensorReducer, ForwardPhase.DECODE);
    }

    @Override
    public TensorRef forward(TensorRef input, Optional<Consumer<List<TensorRef>>> tensorReducer,
            ForwardPhase phase) {
        int batchSize = (int) input.shape().first();
        TensorRef projected = model.makeDenseTensorRef(batchSize, hiddenLength * 2);
        TensorRef hidden = model.makeDenseTensorRef(batchSize, hiddenLength);
        TensorRef output = model.makeDenseTensorRef(batchSize, embeddingLength);
        try {
            try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(),
                    "granitemoehybrid.shared_mlp2.input_projection").time()) {
                model.getLighter().batchDotProduct(new BatchDotProduct()
                        .result(projected).a(input).b(inputWeights)
                        .aColumnOffset(0).bColumnOffset(0).columnLength(embeddingLength)
                        .resultRowOffset(0).bRowOffset(0).rowChunkSize(hiddenLength * 2));
            }
            model.emitLayerDebug(layerIndex, "shared_mlp_projected", projected);
            try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(),
                    "granitemoehybrid.shared_mlp2.activation").time()) {
                for (int row = 0; row < batchSize; row++) {
                    for (int column = 0; column < hiddenLength; column++) {
                        float gate = projected.get(row, column);
                        float value = projected.get(row, hiddenLength + column);
                        hidden.set(ActivationFunction.eval(model.getConfig().activationFunction, gate) * value,
                                row, column);
                    }
                }
            }
            model.emitLayerDebug(layerIndex, "shared_mlp_hidden", hidden);
            tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(hidden)));
            try (TensorRef hiddenQuantized = model.getLighter().reshape(hidden, model.getWorkingQType());
                 Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(),
                         "granitemoehybrid.shared_mlp2.output_projection").time()) {
                model.getLighter().batchDotProduct(new BatchDotProduct()
                        .result(output).a(hiddenQuantized).b(outputWeights)
                        .aColumnOffset(0).bColumnOffset(0).columnLength(hiddenLength)
                        .resultRowOffset(0).bRowOffset(0).rowChunkSize(embeddingLength));
            }
            return output;
        } catch (RuntimeException | Error e) {
            output.close();
            throw e;
        } finally {
            projected.close();
            hidden.close();
        }
    }
}
