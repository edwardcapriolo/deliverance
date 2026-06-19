package io.teknek.deliverance.model;

import com.codahale.metrics.Timer;
import com.google.common.base.Preconditions;
import io.teknek.deliverance.generator.FeedForward;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.VectorMath;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorDisplayUtil;
import io.teknek.deliverance.tensor.VectorTensorMathUtils;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import net.jafama.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

import static io.teknek.deliverance.tensor.VectorTensorMathUtils.softMax;

public class MixtureOfExpertsBlock implements FeedForward {

    private static final Logger LOG = LoggerFactory.getLogger(MixtureOfExpertsBlock.class);

    private final AbstractModel model;
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
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), "mixtureofexpertsblock.forward").time()) {
        int batchSize = lnemb.shape().first();
        int hiddenLength = model.config.hiddenLength;
        AbstractTensor result = model.makeTensor(batchSize, model.config.embeddingLength);

        try (AbstractTensor buf = model.makeTensor(1, hiddenLength);
             AbstractTensor buf2 = model.makeTensor(1, hiddenLength);
             AbstractTensor moeResult = model.makeTensor(1, model.config.embeddingLength)) {

            for (int b = 0; b < batchSize; b++) {
                AbstractTensor lnembSlice = lnemb.slice(true, b);
                try (Timer.Context ignoredRouter = InferenceProfiler.timer(model.getMetricRegistry(), "mixtureofexpertsblock.router_projection").time()) {
                    // Apply each experts gate to the input.
                    VectorMath.pfor(0, numberOfExperts, i -> {
                        expertResults.set(
                                model.configurableTensorProvider.get().dotProduct(lnembSlice,
                                        moeGateWeight.slice(true, i), 0, 0, model.config.embeddingLength),
                                0, i);
                    }, model.getPool());
                }

                try (Timer.Context ignoredSoftmax = InferenceProfiler.timer(model.getMetricRegistry(), "mixtureofexpertsblock.router_softmax").time()) {
                    VectorTensorMathUtils.softMax(expertResults,0, numberOfExperts);
                }

                try (Timer.Context ignoredTopk = InferenceProfiler.timer(model.getMetricRegistry(), "mixtureofexpertsblock.router_topk").time()) {
                    topk(expertResults);
                }
                if (LOG.isDebugEnabled()) {
                    LOG.debug("Experts after softmax: {} selected: {}", TensorDisplayUtil.pretty2dDisplayAll(expertResults),
                            Arrays.toString(selectedExperts));
                }

                // Re-normalize the selected top-k gate weights to sum to 1
                float gateWeightSum = 0.0f;
                for (int i = 0; i < numberOfExpertsPerToken; i++) {
                    gateWeightSum += expertResults.get(0, selectedExperts[i]);
                }

                for (int i = 0; i < numberOfExpertsPerToken; i++) {
                    if (InferenceProfiler.isEnabled()) {
                        InferenceProfiler.counter(model.getMetricRegistry(),
                                "mixtureofexpertsblock.selected_expert." + selectedExperts[i]).inc();
                    }
                    try (Timer.Context ignoredExpert = InferenceProfiler.timer(model.getMetricRegistry(), "mixtureofexpertsblock.expert_forward").time()) {
                    float gateWeight = expertResults.get(0, selectedExperts[i]) / gateWeightSum;
                    batchWeights[0] = fullyConnectedWeights[selectedExperts[i]];
                    batchWeights[1] = upProjectionWeights[selectedExperts[i]];
                    AbstractTensor projectionWeight = projectionWeights[selectedExperts[i]];
                    batchResults[0] = buf;
                    batchResults[1] = buf2;

                    try (Timer.Context ignoredGateUp = InferenceProfiler.timer(model.getMetricRegistry(), "mixtureofexpertsblock.expert_gate_up_projection").time()) {
                        VectorMath.pchunk(0, hiddenLength, (chunkStart, chunkSize) -> {
                            model.configurableTensorProvider.get()
                                    .dotProductBatchChunk(
                                            batchResults,
                                            lnembSlice,
                                            batchWeights,
                                            0,
                                            model.config.embeddingLength,
                                            chunkStart,
                                            chunkSize);
                        }, model.configurableTensorProvider.get().parallelSplitSize(), model.getPool());
                    }

                    try (Timer.Context ignoredActivation = InferenceProfiler.timer(model.getMetricRegistry(), "mixtureofexpertsblock.expert_activation_multiply").time()) {
                        VectorMath.pfor(0, hiddenLength, iv -> {
                            float w1 = buf.get(0, iv);
                            float w1a = ActivationFunction.eval(activationFunction, w1);
                            //buf.set(w1a, 0, iv);
                            buf.set(w1a * buf2.get(0, iv),0, iv);
                        }, model.getPool());
                    }

                    //model.configurableTensorProvider.get().maccumulate(buf, buf2, dctx.hiddenSegmentStart, dctx.hiddenSegmentLength);

                    tensorReducer.ifPresent(func -> func.accept(Collections.singletonList(buf)));

                    // matmul the projection and scale by gate weight

                    //model.configurableTensorProvider.get().scale(0.0f, moeResult, 0 , model.config.embeddingLength);
                    if (InferenceProfiler.isEnabled()) {
                        InferenceProfiler.counter(model.getMetricRegistry(), "mixtureofexpertsblock.down_quantize.input_dtype." + buf.dType()).inc();
                    }
                    try (AbstractTensor bufq = downQuantize(buf)) {
                        try (Timer.Context ignoredDown = InferenceProfiler.timer(model.getMetricRegistry(), "mixtureofexpertsblock.expert_down_projection").time()) {
                        VectorMath.pchunk(0, model.config.embeddingLength, (chunkStart, chunkSize) -> {
                            model.configurableTensorProvider.get()
                                    .dotProductChunk(moeResult, bufq, projectionWeight, 0, hiddenLength, chunkStart, chunkSize);
                        }, model.configurableTensorProvider.get().parallelSplitSize(), model.getPool());
                        }
                    }

                    try (Timer.Context ignoredScale = InferenceProfiler.timer(model.getMetricRegistry(), "mixtureofexpertsblock.expert_scale_accumulate").time()) {
                        model.configurableTensorProvider.get().scale(gateWeight, moeResult, 0, model.config.embeddingLength);

                        if (i == 0) {
                            result.slice(b).copyFrom(moeResult, 0,0, model.config.embeddingLength);
                        } else {
                            model.configurableTensorProvider.get().accumulate(result.slice(b), moeResult, 0, model.config.embeddingLength);
                        }
                    }
                    }
                }
            }

            return result;
        }
        }
    }

    private AbstractTensor downQuantize(AbstractTensor buf) {
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), "mixtureofexpertsblock.down_quantize").time()) {
            return model.maybeQuantize(buf);
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

}
