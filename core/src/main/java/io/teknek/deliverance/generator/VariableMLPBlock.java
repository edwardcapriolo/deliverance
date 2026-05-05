package io.teknek.deliverance.generator;

import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.stream.IntStream;

public class VariableMLPBlock implements FeedForward {
    private final AbstractModel model;
    private final ActivationFunction.Type activationFunction;
    private final AbstractTensor fullyConnectedWeights;
    private final AbstractTensor projectionWeights;
    private final AbstractTensor upProjectionWeights;
    private final int hiddenLength;
    private final AbstractTensor[] batchResults = new AbstractTensor[2];
    private final AbstractTensor[] batchWeights = new AbstractTensor[2];
    private final ConfigurableTensorProvider configurableTensorProvider;

    public VariableMLPBlock(
            AbstractModel model,
            ActivationFunction.Type activationFunction,
            AbstractTensor fullyConnectedWeights,
            AbstractTensor projectionWeights,
            AbstractTensor upProjectionWeights,
            int hiddenLength,
            ConfigurableTensorProvider configurableTensorProvider
    ) {
        this.model = model;
        this.activationFunction = activationFunction;
        this.fullyConnectedWeights = fullyConnectedWeights;
        this.projectionWeights = projectionWeights;
        this.upProjectionWeights = upProjectionWeights;
        this.hiddenLength = hiddenLength;
        this.configurableTensorProvider = configurableTensorProvider;
        this.batchWeights[0] = fullyConnectedWeights;
        this.batchWeights[1] = upProjectionWeights;

        configurableTensorProvider.get().registerModelTensor(fullyConnectedWeights);
        configurableTensorProvider.get().registerModelTensor(upProjectionWeights);
        configurableTensorProvider.get().registerModelTensor(projectionWeights);
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        int batchSize = input.shape().first();
        try (
                AbstractTensor gate = model.getTensorAllocator().getDirty(model.getWorkingDType(), TensorShape.of(batchSize, hiddenLength));
                AbstractTensor up = model.getTensorAllocator().getDirty(model.getWorkingDType(), TensorShape.of(batchSize, hiddenLength))
        ) {
            batchResults[0] = gate;
            batchResults[1] = up;
            VectorMath.pchunk(0, hiddenLength, (chunkStart, chunkSize) -> configurableTensorProvider.get()
                    .dotProductBatchChunk(batchResults, input, batchWeights, 0, model.getConfig().embeddingLength, chunkStart, chunkSize),
                    configurableTensorProvider.get().parallelSplitSize(), model.getPool());

            IntStream.range(0, hiddenLength).parallel().forEach(i -> {
                for (int j = 0; j < batchSize; j++) {
                    float activated = ActivationFunction.eval(activationFunction, gate.get(j, i));
                    gate.set(activated * up.get(j, i), j, i);
                }
            });

            try (AbstractTensor gateQ = model.maybeQuantize(gate)) {
                AbstractTensor result = model.makeTensor(batchSize, model.getConfig().embeddingLength);
                VectorMath.pchunk(0, model.getConfig().embeddingLength, (chunkStart, chunkSize) -> configurableTensorProvider.get()
                        .dotProductChunk(result, gateQ, projectionWeights, 0, hiddenLength, chunkStart, chunkSize),
                        configurableTensorProvider.get().parallelSplitSize(), model.getPool());
                tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(result)));
                return result;
            }
        }
    }
}
