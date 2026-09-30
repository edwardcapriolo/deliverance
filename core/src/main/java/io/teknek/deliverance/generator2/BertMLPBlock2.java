package io.teknek.deliverance.generator2;

import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** TensorRef BERT feed-forward block with bias-bearing GELU projections. */
public final class BertMLPBlock2 implements FeedForward2 {
    private final AbstractModel model;
    private final TensorRef intermediateWeight;
    private final TensorRef outputWeight;
    private final TensorRef intermediateBias;
    private final TensorRef outputBias;
    private final Lighter lighter;
    private final int intermediateLength;
    private final int embeddingLength;

    public BertMLPBlock2(AbstractModel model, TensorRef intermediateWeight, TensorRef outputWeight,
            TensorRef intermediateBias, TensorRef outputBias, Lighter lighter) {
        this.model = model;
        this.intermediateWeight = intermediateWeight;
        this.outputWeight = outputWeight;
        this.intermediateBias = intermediateBias;
        this.outputBias = outputBias;
        this.lighter = lighter;
        this.intermediateLength = model.getConfig().hiddenLength;
        this.embeddingLength = model.getConfig().embeddingLength;
    }

    @Override
    public TensorRef forward(TensorRef input, Optional<Consumer<List<TensorRef>>> tensorReducer) {
        return forward(input, tensorReducer, ForwardPhase.DECODE);
    }

    @Override
    public TensorRef forward(TensorRef input, Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase) {
        int rows = (int) input.shape().first();
        TensorRef intermediate = model.makeDenseTensorRef(rows, intermediateLength);
        TensorRef output = model.makeDenseTensorRef(rows, embeddingLength);
        try {
            project(intermediate, input, intermediateWeight, intermediateBias);
            activate(intermediate);
            project(output, intermediate, outputWeight, outputBias);
            tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(output)));
            return output;
        } catch (RuntimeException | Error e) {
            output.close();
            throw e;
        } finally {
            intermediate.close();
        }
    }

    private void project(TensorRef output, TensorRef input, TensorRef weight, TensorRef bias) {
        int outputLength = (int) output.shape().last();
        int inputLength = (int) input.shape().last();
        model.runChunks("bertmlp2.projection", 0, outputLength,
                model.primaryTensorOperations().parallelSplitSize(), Optional.empty(),
                (chunkStart, chunkSize) -> lighter.dotProductRows(output, input, weight, 0,
                        inputLength, chunkStart, chunkSize, chunkStart));
        for (int row = 0; row < output.shape().first(); row++) {
            for (int column = 0; column < outputLength; column++) {
                output.set(output.get(row, column) + bias.get(0, column), row, column);
            }
        }
    }

    private void activate(TensorRef tensor) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.set(ActivationFunction.eval(model.getConfig().activationFunction,
                        tensor.get(row, column)), row, column);
            }
        }
    }
}
