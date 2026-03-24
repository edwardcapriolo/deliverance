package io.teknek.deliverance.model;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.generator.FeedForward;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.BiIntConsumer;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.safetensors.DistributedContext;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import net.jafama.FastMath;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

import static io.teknek.deliverance.tensor.VectorTensorMathUtils.softMax;

public class MixtureOfExpertsBlock implements FeedForward {

    private final AbstractModel model;
    private final DistributedContext dctx;
    private final AbstractTensor moeGateWeight;
    private final int numberOfExperts;
    private final int numberOfExpertsPerToken;
    private final AbstractTensor fullyConnectedWeights [];
    private final AbstractTensor projectionWeights [];
    private final AbstractTensor upProjectionWeights [];
    private final FloatBufferTensor expertResults;
    private final int [] selectedExperts;
    private final ActivationFunction.Type activationFunction;

    private final AbstractTensor[] batchResults;
    private final AbstractTensor[] batchWeights;

    public MixtureOfExpertsBlock (
            AbstractModel model,
            int numberOfExperts,
            int numberOfExpertsPerToken,
            ActivationFunction.Type activationFunction,
            AbstractTensor moeGateWeight,
            AbstractTensor[] fullyConnectedWeights,
            AbstractTensor[] projectionWeights,
            AbstractTensor[] upProjectionWeights) {
        this.model = model;
        this.dctx = model.config.dctx();
        this.numberOfExperts = numberOfExperts;
        this.numberOfExpertsPerToken = numberOfExpertsPerToken;
        this.moeGateWeight = moeGateWeight;
        this.activationFunction = activationFunction;
        this.fullyConnectedWeights = fullyConnectedWeights;
        this.projectionWeights = projectionWeights;
        this.upProjectionWeights = upProjectionWeights;
        this.expertResults = new FloatBufferTensor(numberOfExperts);
        this.selectedExperts = new int[numberOfExpertsPerToken];
        this.batchResults = new AbstractTensor[2];
        this.batchWeights = new AbstractTensor[2];
    }

    @Override
    public AbstractTensor forward(AbstractTensor lnemb, Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        int batchSize = lnemb.shape().first();
        int hiddenLength = model.config.hiddenLength;
        AbstractTensor result = model.makeTensor(batchSize, model.config.embeddingLength);

        try (AbstractTensor buf = model.makeTensor(1, hiddenLength);
             AbstractTensor buf2 = model.makeTensor(1, hiddenLength);
             AbstractTensor moeResult = model.makeTensor(1, model.config.embeddingLength)) {

            for (int b = 0; b < batchSize; b++) {
                AbstractTensor lnembSlice = lnemb.slice(true, b);
                // Apply each experts gate to the input
                VectorMath.pfor(0, numberOfExperts, i -> {
                    expertResults.set(
                            model.configurableTensorProvider.get().dotProduct(lnembSlice,
                                    moeGateWeight.slice(true, i), 0, 0, model.config.embeddingLength),
                            0, i);
                });

                softMax(expertResults, 0, numberOfExperts);
                topk(expertResults);

                for (int i = 0; i < numberOfExpertsPerToken; i++) {
                    batchWeights[0] = fullyConnectedWeights[selectedExperts[i]];
                    batchWeights[1] = upProjectionWeights[selectedExperts[i]];
                    AbstractTensor projectionWeight = projectionWeights[selectedExperts[i]];
                    batchResults[0] = buf;
                    batchResults[1] = buf2;

                    VectorMath.pchunk(dctx.hiddenSegmentStart, dctx.hiddenSegmentLength, (chunkStart, chunkSize) -> {
                        model.configurableTensorProvider.get()
                                .dotProductBatchChunk(
                                        batchResults,
                                        lnembSlice,
                                        batchWeights,
                                        0,
                                        model.config.embeddingLength,
                                        chunkStart,
                                        chunkSize);
                    }, model.configurableTensorProvider.get().parallelSplitSize());

                    VectorMath.pfor(dctx.hiddenSegmentStart, dctx.hiddenSegmentEnd, iv -> {
                        float w1 = buf.get(0, iv);
                        float w1a = ActivationFunction.eval(activationFunction, w1);
                        buf.set(w1a, 0, iv);
                    });

                    model.configurableTensorProvider.get().maccumulate(buf, buf2, dctx.hiddenSegmentStart, dctx.hiddenSegmentLength);

                    tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(buf)));

                    // matmul the projection and sum into result
                    try (AbstractTensor bufq = model.maybeQuantize(buf)) {
                        VectorMath.pchunk(0, model.config.embeddingLength, (chunkStart, chunkSize) -> {
                            model.configurableTensorProvider.get()
                                    .dotProductChunk(moeResult, bufq, projectionWeight, 0, hiddenLength, chunkStart, chunkSize);
                        }, model.configurableTensorProvider.get().parallelSplitSize());
                    }

                    if (i == 0) {
                        result.copyFrom(moeResult, 0, 0, model.config.embeddingLength);
                    } else {
                        model.configurableTensorProvider.get().accumulate(result.slice(b), moeResult, 0, model.config.embeddingLength);
                    }
                }
            }

            return result;
        }

    }



    private int[] topk(FloatBufferTensor probs) {
        long length = probs.size();
        for (int i = 0; i < numberOfExpertsPerToken; i++) {
            selectedExperts[i] = i;
        }
        for (int i = numberOfExpertsPerToken; i < length; i++) {
            int min = 0;
            for (int j = 1; j < numberOfExpertsPerToken; j++) {
                if (probs.get(0, selectedExperts[j]) < probs.get(0, selectedExperts[min])) {
                    min = j;
                }
            }
            if (probs.get(0, i) > probs.get(0, selectedExperts[min])) {
                selectedExperts[min] = i;
            }
        }
        return selectedExperts;
    }

    public static void softMax(AbstractTensor x, int offset, int length) {
        Preconditions.checkArgument(x.shape().first() == 1);
        long size = offset + length;

        float max_val = x.get(0, offset);
        for (int i = offset + 1; i < size; i++) {
            if (x.get(0, i) > max_val) {
                max_val = x.get(0, i);
            }
        }
        float sum = 0.0f;
        for (int i = offset; i < size; i++) {
            x.set((float) FastMath.exp(x.get(0, i) - max_val), 0, i);
            sum += x.get(0, i);
        }
        for (int i = 0; i < size; i++) {
            x.set(x.get(0, i) / sum, 0, i);
        }
    }
}
