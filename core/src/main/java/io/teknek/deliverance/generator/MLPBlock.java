package io.teknek.deliverance.generator;

import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.safetensors.DistributedContext;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.stream.IntStream;

/**
 * A standard Multi Layer Perceptron block for Transformer models
 */
public class MLPBlock implements FeedForward {
    private final AbstractModel model;
    private final DistributedContext dctx;
    private final Optional<AbstractTensor> fullyConnectedBias;
    private final AbstractTensor fullyConnectedWeights;

    private final Optional<AbstractTensor> projectionBias;
    private final AbstractTensor projectionWeights;

    private final AbstractTensor upProjectionWeights;

    private final ActivationFunction.Type activationFunction;

    private final AbstractTensor[] batchResults;
    private final AbstractTensor[] batchWeights;
    private final ConfigurableTensorProvider configurableTensorProvider;

    public MLPBlock(
            AbstractModel model,
            ActivationFunction.Type activationFunction,
            AbstractTensor fullyConnectedWeights,
            AbstractTensor projectionWeights,
            AbstractTensor upProjectionWeights,
            ConfigurableTensorProvider configurableTensorProvider
    ) {
        this(model, activationFunction, Optional.empty(), fullyConnectedWeights,
                Optional.empty(), projectionWeights, upProjectionWeights, configurableTensorProvider);
    }

    public MLPBlock(
            AbstractModel model,
            ActivationFunction.Type activationFunction,
            Optional<AbstractTensor> fullyConnectedBias,
            AbstractTensor fullyConnectedWeights,
            Optional<AbstractTensor> projectionBias,
            AbstractTensor projectionWeights,
            AbstractTensor upProjectionWeights,
            ConfigurableTensorProvider configurableTensorProvider
    ) {
        this.model = model;
        this.dctx = model.getConfig().dctx();
        this.activationFunction = activationFunction;
        this.fullyConnectedBias = fullyConnectedBias;
        this.fullyConnectedWeights = fullyConnectedWeights;
        this.projectionBias = projectionBias;
        this.projectionWeights = projectionWeights;
        this.upProjectionWeights = upProjectionWeights;
        this.batchResults = new AbstractTensor[2];
        this.batchWeights = new AbstractTensor[] { fullyConnectedWeights, upProjectionWeights };
        this.configurableTensorProvider = configurableTensorProvider;

        configurableTensorProvider.get().registerModelTensor(fullyConnectedWeights);
        if (upProjectionWeights != null) {
            configurableTensorProvider.get().registerModelTensor(upProjectionWeights);
        }
        configurableTensorProvider.get().registerModelTensor(projectionWeights);
    }

    // For FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    @Override
    public AbstractTensor forward(AbstractTensor lnemb, Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        int hiddenLength = model.getConfig().hiddenLength;
        int batchSize = lnemb.shape().first();
        try (
                //TODO ensure its ok to use dirty here do we write the entire tensor
                AbstractTensor buf = model.getTensorCache().getDirty(model.getWorkingDType(), TensorShape.of(batchSize, hiddenLength));
                AbstractTensor buf2 = model.getTensorCache().getDirty(model.getWorkingDType(), TensorShape.of(batchSize, hiddenLength));
        ) {

            batchResults[0] = buf;
            batchResults[1] = buf2;

            VectorMath.pchunk(dctx.hiddenSegmentStart, dctx.hiddenSegmentLength, (chunkStart, chunkSize) -> {
                if (upProjectionWeights != null) {
                    configurableTensorProvider.get()
                            .dotProductBatchChunk(batchResults, lnemb, batchWeights, 0, model.getConfig().embeddingLength, chunkStart, chunkSize);
                } else {
                    configurableTensorProvider.get()
                            .dotProductChunk(buf, lnemb, fullyConnectedWeights, 0, model.getConfig().embeddingLength, chunkStart, chunkSize);
                }
            }, configurableTensorProvider.get().parallelSplitSize());

            fullyConnectedBias.ifPresent(
                    bias -> configurableTensorProvider.get().accumulate(buf, bias, dctx.hiddenSegmentStart, dctx.hiddenSegmentLength)
            );

            // Not using pfor because we can use all cores
            IntStream.range(dctx.hiddenSegmentStart, dctx.hiddenSegmentEnd).parallel().forEach(i -> {
                for (int j = 0; j < batchSize; j++) {
                    float w1 = buf.get(j, i);
                    float w1a = ActivationFunction.eval(activationFunction, w1);
                    buf.set(w1a, j, i);
                }
            });

            if (upProjectionWeights != null) {
                configurableTensorProvider.get().maccumulate(buf, buf2, 0, hiddenLength);
            }

            try (AbstractTensor bufq = model.maybeQuantize(buf)) {
                // matmul the projection and sum into input
                AbstractTensor result = model.makeTensor(batchSize, model.getConfig().embeddingLength);
                VectorMath.pchunk(0, model.getConfig().embeddingLength, (chunkStart, chunkSize) -> {
                    configurableTensorProvider.get()
                            .dotProductChunk(
                                    result,
                                    bufq,
                                    projectionWeights,
                                    dctx.hiddenSegmentStart,
                                    dctx.hiddenSegmentLength,
                                    chunkStart,
                                    chunkSize
                            );
                }, configurableTensorProvider.get().parallelSplitSize());

                tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(result)));

                projectionBias.ifPresent(bias -> configurableTensorProvider.get().accumulate(result, bias, 0, model.getConfig().embeddingLength));
                return result;
            }
        }
    }
}
