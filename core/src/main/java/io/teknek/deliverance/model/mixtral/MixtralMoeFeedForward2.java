package io.teknek.deliverance.model.mixtral;

import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.generator2.FeedForward2;
import io.teknek.deliverance.tensor2.BatchDotProduct;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** TensorRef-backed Mixtral router and expert feed-forward path. */
final class MixtralMoeFeedForward2 implements FeedForward2 {
    private final AbstractModel model;
    private final MixtralConfig config;
    private final TensorRef routerWeights;
    private final TensorRef[] gateWeights;
    private final TensorRef[] upWeights;
    private final TensorRef[] downWeights;
    private final Lighter lighter;

    MixtralMoeFeedForward2(AbstractModel model, MixtralConfig config, TensorRef routerWeights,
            TensorRef[] gateWeights, TensorRef[] upWeights, TensorRef[] downWeights, Lighter lighter) {
        this.model = model;
        this.config = config;
        this.routerWeights = routerWeights;
        this.gateWeights = gateWeights;
        this.upWeights = upWeights;
        this.downWeights = downWeights;
        this.lighter = lighter;
    }

    @Override
    public TensorRef forward(TensorRef input, Optional<Consumer<List<TensorRef>>> tensorReducer) {
        return forward(input, tensorReducer, ForwardPhase.DECODE);
    }

    @Override
    public TensorRef forward(TensorRef input, Optional<Consumer<List<TensorRef>>> tensorReducer,
            ForwardPhase phase) {
        int batchSize = (int) input.shape().first();
        TensorRef output = model.makeDenseTensorRef(batchSize, config.embeddingLength);
        TensorRef logits = model.makeDenseTensorRef(batchSize, config.numberOfExperts);
        lighter.clear(output);
        try {
            lighter.batchDotProduct(new BatchDotProduct().result(logits).a(input).b(routerWeights)
                    .aColumnOffset(0).bColumnOffset(0).columnLength(config.embeddingLength)
                    .resultRowOffset(0).bRowOffset(0).rowChunkSize(config.numberOfExperts));
            int[] selected = new int[batchSize * config.numberOfExpertsPerToken];
            float[] weights = new float[selected.length];
            route(logits, selected, weights);
            for (int expert = 0; expert < config.numberOfExperts; expert++) {
                int count = count(selected, expert);
                if (count != 0) runExpert(input, output, selected, weights, expert, count, phase);
            }
            tensorReducer.ifPresent(reducer -> reducer.accept(Collections.singletonList(output)));
            return output;
        } catch (RuntimeException | Error failure) {
            output.close();
            throw failure;
        } finally {
            logits.close();
        }
    }

    private void runExpert(TensorRef input, TensorRef output, int[] selected, float[] weights,
            int expert, int tokenCount, ForwardPhase phase) {
        TensorRef expertInput = model.makeDenseTensorRef(tokenCount, config.embeddingLength);
        TensorRef gate = model.makeDenseTensorRef(tokenCount, config.hiddenLength);
        TensorRef up = model.makeDenseTensorRef(tokenCount, config.hiddenLength);
        TensorRef down = model.makeDenseTensorRef(tokenCount, config.embeddingLength);
        int[] sourceRows = new int[tokenCount];
        float[] routeWeights = new float[tokenCount];
        try {
            gather(input, expertInput, selected, weights, expert, sourceRows, routeWeights);
            project(expertInput, gate, gateWeights[expert], config.embeddingLength);
            project(expertInput, up, upWeights[expert], config.embeddingLength);
            for (int row = 0; row < tokenCount; row++) {
                for (int column = 0; column < config.hiddenLength; column++) {
                    gate.set(ActivationFunction.eval(config.activationFunction, gate.get(row, column))
                            * up.get(row, column), row, column);
                }
            }
            project(gate, down, downWeights[expert], config.hiddenLength);
            for (int row = 0; row < tokenCount; row++) {
                int outputRow = sourceRows[row];
                float routeWeight = routeWeights[row];
                for (int column = 0; column < config.embeddingLength; column++) {
                    output.set(output.get(outputRow, column) + down.get(row, column) * routeWeight,
                            outputRow, column);
                }
            }
        } finally {
            expertInput.close();
            gate.close();
            up.close();
            down.close();
        }
    }

    private void project(TensorRef input, TensorRef output, TensorRef weights, int inputLength) {
        lighter.batchDotProduct(new BatchDotProduct().result(output).a(input).b(weights)
                .aColumnOffset(0).bColumnOffset(0).columnLength(inputLength)
                .resultRowOffset(0).bRowOffset(0).rowChunkSize((int) output.shape().last()));
    }

    private void gather(TensorRef input, TensorRef destination, int[] selected, float[] weights,
            int expert, int[] sourceRows, float[] routeWeights) {
        int destinationRow = 0;
        for (int row = 0; row < input.shape().first(); row++) {
            for (int top = 0; top < config.numberOfExpertsPerToken; top++) {
                int index = row * config.numberOfExpertsPerToken + top;
                if (selected[index] == expert) {
                    lighter.copy(input, input.shape().getOffset(row, 0), destination,
                            destination.shape().getOffset(destinationRow, 0), config.embeddingLength);
                    sourceRows[destinationRow] = row;
                    routeWeights[destinationRow] = weights[index];
                    destinationRow++;
                }
            }
        }
    }

    private void route(TensorRef logits, int[] selected, float[] weights) {
        for (int row = 0; row < logits.shape().first(); row++) {
            float max = Float.NEGATIVE_INFINITY;
            float[] probabilities = new float[config.numberOfExperts];
            for (int expert = 0; expert < config.numberOfExperts; expert++) {
                probabilities[expert] = logits.get(row, expert);
                max = Math.max(max, probabilities[expert]);
            }
            float sum = 0.0f;
            for (int expert = 0; expert < config.numberOfExperts; expert++) {
                probabilities[expert] = (float) Math.exp(probabilities[expert] - max);
                sum += probabilities[expert];
            }
            for (int expert = 0; expert < config.numberOfExperts; expert++) probabilities[expert] /= sum;
            int[] topExperts = new int[config.numberOfExpertsPerToken];
            float[] topWeights = new float[config.numberOfExpertsPerToken];
            Arrays.fill(topExperts, -1);
            Arrays.fill(topWeights, Float.NEGATIVE_INFINITY);
            for (int expert = 0; expert < config.numberOfExperts; expert++) {
                int insert = config.numberOfExpertsPerToken - 1;
                if (probabilities[expert] <= topWeights[insert]) continue;
                while (insert > 0 && probabilities[expert] > topWeights[insert - 1]) {
                    topWeights[insert] = topWeights[insert - 1];
                    topExperts[insert] = topExperts[insert - 1];
                    insert--;
                }
                topWeights[insert] = probabilities[expert];
                topExperts[insert] = expert;
            }
            float topSum = 0.0f;
            for (float weight : topWeights) topSum += weight;
            int base = row * config.numberOfExpertsPerToken;
            for (int top = 0; top < topWeights.length; top++) {
                selected[base + top] = topExperts[top];
                weights[base + top] = topWeights[top] / topSum;
            }
        }
    }

    private int count(int[] selected, int expert) {
        int count = 0;
        for (int value : selected) if (value == expert) count++;
        return count;
    }
}
