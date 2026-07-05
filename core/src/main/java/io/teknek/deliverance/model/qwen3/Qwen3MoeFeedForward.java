package io.teknek.deliverance.model.qwen3;

import com.codahale.metrics.Timer;
import io.teknek.deliverance.generator.FeedForward;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import net.jafama.FastMath;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** Qwen3 MoE sparse MLP, matching HF {@code Qwen3MoeSparseMoeBlock}. */
public class Qwen3MoeFeedForward implements FeedForward {
    private final AbstractModel model;
    private final Qwen3MoeConfig config;
    private final AbstractTensor routerWeights;
    private final AbstractTensor[] expertGateWeights;
    private final AbstractTensor[] expertUpWeights;
    private final AbstractTensor[] expertDownWeights;
    private final ConfigurableTensorProvider configurableTensorProvider;

    public Qwen3MoeFeedForward(AbstractModel model, Qwen3MoeConfig config, AbstractTensor routerWeights,
            AbstractTensor[] expertGateWeights, AbstractTensor[] expertUpWeights, AbstractTensor[] expertDownWeights,
            ConfigurableTensorProvider configurableTensorProvider) {
        this.model = model;
        this.config = config;
        this.routerWeights = routerWeights;
        this.expertGateWeights = expertGateWeights;
        this.expertUpWeights = expertUpWeights;
        this.expertDownWeights = expertDownWeights;
        this.configurableTensorProvider = configurableTensorProvider;
        configurableTensorProvider.get().registerModelTensor(routerWeights);
        for (int i = 0; i < config.numExperts; i++) {
            configurableTensorProvider.get().registerModelTensor(expertGateWeights[i]);
            configurableTensorProvider.get().registerModelTensor(expertUpWeights[i]);
            configurableTensorProvider.get().registerModelTensor(expertDownWeights[i]);
        }
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), "qwen3moe.forward").time()) {
            int batchSize = input.shape().first();
            AbstractTensor output = model.makeTensor(batchSize, config.embeddingLength);
            int[] selectedExperts = new int[config.numExpertsPerToken];
            float[] selectedWeights = new float[config.numExpertsPerToken];
            for (int batch = 0; batch < batchSize; batch++) {
                route(input, batch, selectedExperts, selectedWeights);
                for (int top = 0; top < config.numExpertsPerToken; top++) {
                    accumulateExpert(input, output, batch, selectedExperts[top], selectedWeights[top]);
                }
            }
            return output;
        }
    }

    private void route(AbstractTensor input, int batch, int[] selectedExperts, float[] selectedWeights) {
        float[] probabilities = new float[config.numExperts];
        float max = Float.NEGATIVE_INFINITY;
        for (int expert = 0; expert < config.numExperts; expert++) {
            float score = 0.0f;
            for (int hidden = 0; hidden < config.embeddingLength; hidden++) {
                score += input.get(batch, hidden) * routerWeights.get(expert, hidden);
            }
            probabilities[expert] = score;
            max = Math.max(max, score);
        }
        double sum = 0.0d;
        for (int expert = 0; expert < config.numExperts; expert++) {
            float value = (float) FastMath.exp(probabilities[expert] - max);
            probabilities[expert] = value;
            sum += value;
        }
        for (int expert = 0; expert < config.numExperts; expert++) {
            probabilities[expert] /= (float) sum;
        }
        topK(probabilities, selectedExperts, selectedWeights);
        if (config.normTopkProb) {
            float selectedSum = 0.0f;
            for (float selectedWeight : selectedWeights) {
                selectedSum += selectedWeight;
            }
            if (selectedSum != 0.0f) {
                for (int i = 0; i < selectedWeights.length; i++) {
                    selectedWeights[i] /= selectedSum;
                }
            }
        }
    }

    private void topK(float[] probabilities, int[] selectedExperts, float[] selectedWeights) {
        for (int i = 0; i < selectedExperts.length; i++) {
            selectedExperts[i] = i;
            selectedWeights[i] = probabilities[i];
        }
        for (int expert = selectedExperts.length; expert < probabilities.length; expert++) {
            int min = 0;
            for (int i = 1; i < selectedExperts.length; i++) {
                if (selectedWeights[i] < selectedWeights[min]) {
                    min = i;
                }
            }
            if (probabilities[expert] > selectedWeights[min]) {
                selectedExperts[min] = expert;
                selectedWeights[min] = probabilities[expert];
            }
        }
    }

    private void accumulateExpert(AbstractTensor input, AbstractTensor output, int batch, int expert, float routeWeight) {
        for (int hidden = 0; hidden < config.embeddingLength; hidden++) {
            float down = 0.0f;
            for (int intermediate = 0; intermediate < config.moeIntermediateSize; intermediate++) {
                float gate = 0.0f;
                float up = 0.0f;
                for (int inputHidden = 0; inputHidden < config.embeddingLength; inputHidden++) {
                    float value = input.get(batch, inputHidden);
                    gate += value * expertGateWeights[expert].get(intermediate, inputHidden);
                    up += value * expertUpWeights[expert].get(intermediate, inputHidden);
                }
                down += ActivationFunction.eval(config.activationFunction, gate)
                        * up
                        * expertDownWeights[expert].get(hidden, intermediate);
            }
            output.set(output.get(batch, hidden) + down * routeWeight, batch, hidden);
        }
    }
}
