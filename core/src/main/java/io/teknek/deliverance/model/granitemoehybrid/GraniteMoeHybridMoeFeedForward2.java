package io.teknek.deliverance.model.granitemoehybrid;

import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.generator2.FeedForward2;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** TensorRef sparse-expert branch used with Granite's always-present shared MLP. */
final class GraniteMoeHybridMoeFeedForward2 implements FeedForward2 {
    private final AbstractModel model;
    private final GraniteMoeHybridConfig config;
    private final FeedForward2 shared;
    private final TensorRef router, inputWeights, outputWeights;

    GraniteMoeHybridMoeFeedForward2(AbstractModel model, GraniteMoeHybridConfig config, FeedForward2 shared,
            TensorRef router, TensorRef inputWeights, TensorRef outputWeights) {
        this.model = model;
        this.config = config;
        this.shared = shared;
        this.router = router;
        this.inputWeights = inputWeights;
        this.outputWeights = outputWeights;
    }

    @Override
    public TensorRef forward(TensorRef input, Optional<Consumer<List<TensorRef>>> reducer) {
        return forward(input, reducer, ForwardPhase.DECODE);
    }

    @Override
    public TensorRef forward(TensorRef input, Optional<Consumer<List<TensorRef>>> reducer, ForwardPhase phase) {
        int tokens = (int) input.shape().first();
        TensorRef result = shared.forward(input, reducer, phase);
        int[][] experts = new int[tokens][config.numExpertsPerToken];
        float[][] weights = new float[tokens][config.numExpertsPerToken];
        TensorRef logits = model.makeDenseTensorRef(tokens, config.numLocalExperts);
        try {
            model.getLighter().dotProductRows(logits, input, router, 0, config.embeddingLength,
                    0, config.numLocalExperts, 0);
            for (int token = 0; token < tokens; token++) topK(logits, token, experts[token], weights[token]);
            for (int expert = 0; expert < config.numLocalExperts; expert++) {
                List<Integer> rows = new ArrayList<>();
                for (int token = 0; token < tokens; token++) {
                    for (int top = 0; top < config.numExpertsPerToken; top++) {
                        if (experts[token][top] == expert) rows.add(token);
                    }
                }
                if (!rows.isEmpty()) applyExpert(input, result, expert, rows, experts, weights);
            }
            return result;
        } catch (RuntimeException | Error e) {
            result.close();
            throw e;
        } finally {
            logits.close();
        }
    }

    private void applyExpert(TensorRef input, TensorRef result, int expert, List<Integer> rows,
            int[][] selected, float[][] selectedWeights) {
        int count = rows.size();
        try (TensorRef expertInput = model.makeDenseTensorRef(count, config.embeddingLength);
             TensorRef projected = model.makeDenseTensorRef(count, config.hiddenLength * 2);
             TensorRef hidden = model.makeDenseTensorRef(count, config.hiddenLength);
             TensorRef output = model.makeDenseTensorRef(count, config.embeddingLength);
             TensorRef inWeight = inputWeights.slice(expert);
             TensorRef outWeight = outputWeights.slice(expert)) {
            for (int i = 0; i < count; i++) {
                try (TensorRef source = input.slice(rows.get(i))) {
                    model.getLighter().copy(source, 0, expertInput, expertInput.shape().getOffset(i, 0),
                            config.embeddingLength);
                }
            }
            model.getLighter().dotProductRows(projected, expertInput, inWeight, 0, config.embeddingLength,
                    0, config.hiddenLength * 2, 0);
            for (int row = 0; row < count; row++) {
                for (int col = 0; col < config.hiddenLength; col++) {
                    hidden.set(ActivationFunction.eval(config.activationFunction, projected.get(row, col))
                            * projected.get(row, config.hiddenLength + col), row, col);
                }
            }
            model.getLighter().dotProductRows(output, hidden, outWeight, 0, config.hiddenLength,
                    0, config.embeddingLength, 0);
            for (int i = 0; i < count; i++) {
                int token = rows.get(i);
                float weight = 0.0f;
                for (int top = 0; top < config.numExpertsPerToken; top++) {
                    if (selected[token][top] == expert) weight += selectedWeights[token][top];
                }
                for (int col = 0; col < config.embeddingLength; col++) {
                    result.set(result.get(token, col) + weight * output.get(i, col), token, col);
                }
            }
        }
    }

    private void topK(TensorRef logits, int row, int[] selected, float[] weights) {
        Arrays.fill(selected, -1);
        Arrays.fill(weights, Float.NEGATIVE_INFINITY);
        for (int expert = 0; expert < config.numLocalExperts; expert++) {
            float logit = logits.get(row, expert);
            int insert = weights.length - 1;
            if (logit <= weights[insert]) continue;
            while (insert > 0 && logit > weights[insert - 1]) {
                weights[insert] = weights[insert - 1]; selected[insert] = selected[insert - 1]; insert--;
            }
            weights[insert] = logit; selected[insert] = expert;
        }
        float max = Float.NEGATIVE_INFINITY;
        for (float weight : weights) max = Math.max(max, weight);
        float sum = 0.0f;
        for (int i = 0; i < weights.length; i++) { weights[i] = (float) Math.exp(weights[i] - max); sum += weights[i]; }
        for (int i = 0; i < weights.length; i++) weights[i] /= sum;
    }
}
