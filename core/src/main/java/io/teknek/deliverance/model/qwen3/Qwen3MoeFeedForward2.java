package io.teknek.deliverance.model.qwen3;

import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor2.BatchDotProduct;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** TensorRef-backed Qwen3 MoE router and expert feed-forward path. */
final class Qwen3MoeFeedForward2 implements io.teknek.deliverance.generator2.FeedForward2 {
    private final AbstractModel model;
    private final Qwen3MoeConfig config;
    private final TensorRef routerWeights;
    private final TensorRef[] expertGateWeights;
    private final TensorRef[] expertUpWeights;
    private final TensorRef[] expertDownWeights;
    private final Lighter lighter;

    Qwen3MoeFeedForward2(AbstractModel model, Qwen3MoeConfig config, TensorRef routerWeights,
            TensorRef[] expertGateWeights, TensorRef[] expertUpWeights, TensorRef[] expertDownWeights,
            Lighter lighter) {
        this.model = model;
        this.config = config;
        this.routerWeights = routerWeights;
        this.expertGateWeights = expertGateWeights;
        this.expertUpWeights = expertUpWeights;
        this.expertDownWeights = expertDownWeights;
        this.lighter = lighter;
    }

    @Override
    public TensorRef forward(TensorRef input, Optional<Consumer<List<TensorRef>>> tensorReducer) {
        return forward(input, tensorReducer, io.teknek.deliverance.generator.ForwardPhase.DECODE);
    }

    @Override
    public TensorRef forward(TensorRef input, Optional<Consumer<List<TensorRef>>> tensorReducer,
            io.teknek.deliverance.generator.ForwardPhase phase) {
        int batchSize = input.shape().first();
        TensorRef output = model.makeDenseTensorRef(batchSize, config.embeddingLength);
        TensorRef logits = model.makeDenseTensorRef(batchSize, config.numExperts);
        zero(output);
        try {
            lighter.batchDotProduct(new BatchDotProduct()
                    .result(logits).a(input).b(routerWeights)
                    .aColumnOffset(0).bColumnOffset(0).columnLength(config.embeddingLength)
                    .resultRowOffset(0).bRowOffset(0).rowChunkSize(config.numExperts));
            int[] selectedExperts = new int[batchSize * config.numExpertsPerToken];
            float[] selectedWeights = new float[selectedExperts.length];
            route(logits, selectedExperts, selectedWeights);

            for (int expert = 0; expert < config.numExperts; expert++) {
                int tokenCount = count(selectedExperts, expert);
                if (tokenCount == 0) {
                    continue;
                }
                runExpert(input, output, selectedExperts, selectedWeights, expert, tokenCount, phase);
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

    private void runExpert(TensorRef input, TensorRef output, int[] selectedExperts, float[] selectedWeights,
            int expert, int tokenCount, io.teknek.deliverance.generator.ForwardPhase phase) {
        TensorRef expertInput = model.makeDenseTensorRef(tokenCount, config.embeddingLength);
        TensorRef gate = model.makeDenseTensorRef(tokenCount, config.moeIntermediateSize);
        TensorRef up = model.makeDenseTensorRef(tokenCount, config.moeIntermediateSize);
        TensorRef down = model.makeDenseTensorRef(tokenCount, config.embeddingLength);
        int[] sourceRows = new int[tokenCount];
        float[] routeWeights = new float[tokenCount];
        try {
            gatherRows(input, expertInput, selectedExperts, selectedWeights, expert, sourceRows, routeWeights);
            project(expertInput, gate, expertGateWeights[expert], config.embeddingLength,
                    config.moeIntermediateSize, phase);
            project(expertInput, up, expertUpWeights[expert], config.embeddingLength,
                    config.moeIntermediateSize, phase);
            for (int row = 0; row < tokenCount; row++) {
                for (int column = 0; column < config.moeIntermediateSize; column++) {
                    gate.set(ActivationFunction.eval(config.activationFunction, gate.get(row, column))
                            * up.get(row, column), row, column);
                }
            }
            project(gate, down, expertDownWeights[expert], config.moeIntermediateSize,
                    config.embeddingLength, phase);
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

    private void project(TensorRef input, TensorRef output, TensorRef weights, int inputLength, int outputLength,
            io.teknek.deliverance.generator.ForwardPhase phase) {
        lighter.batchDotProduct(new BatchDotProduct()
                .result(output).a(input).b(weights)
                .aColumnOffset(0).bColumnOffset(0).columnLength(inputLength)
                .resultRowOffset(0).bRowOffset(0).rowChunkSize(outputLength));
    }

    private void gatherRows(TensorRef input, TensorRef destination, int[] selectedExperts,
            float[] selectedWeights, int expert, int[] sourceRows, float[] routeWeights) {
        int destinationRow = 0;
        for (int row = 0; row < input.shape().first(); row++) {
            int base = row * config.numExpertsPerToken;
            for (int top = 0; top < config.numExpertsPerToken; top++) {
                int selected = base + top;
                if (selectedExperts[selected] == expert) {
                    lighter.copy(input, input.shape().getOffset(row, 0), destination,
                            destination.shape().getOffset(destinationRow, 0), config.embeddingLength);
                    sourceRows[destinationRow] = row;
                    routeWeights[destinationRow] = selectedWeights[selected];
                    destinationRow++;
                }
            }
        }
    }

    private int count(int[] selectedExperts, int expert) {
        int count = 0;
        for (int selected : selectedExperts) {
            if (selected == expert) {
                count++;
            }
        }
        return count;
    }

    private void route(TensorRef logits, int[] selectedExperts, float[] selectedWeights) {
        for (int row = 0; row < logits.shape().first(); row++) {
            float max = Float.NEGATIVE_INFINITY;
            float[] probabilities = new float[config.numExperts];
            for (int expert = 0; expert < config.numExperts; expert++) {
                float score = logits.get(row, expert);
                probabilities[expert] = score;
                max = Math.max(max, score);
            }
            double sum = 0.0d;
            for (int expert = 0; expert < config.numExperts; expert++) {
                probabilities[expert] = (float) Math.exp(probabilities[expert] - max);
                sum += probabilities[expert];
            }
            for (int expert = 0; expert < config.numExperts; expert++) {
                probabilities[expert] /= (float) sum;
            }
            int[] topExperts = new int[config.numExpertsPerToken];
            float[] topWeights = new float[config.numExpertsPerToken];
            Arrays.fill(topExperts, -1);
            Arrays.fill(topWeights, Float.NEGATIVE_INFINITY);
            for (int expert = 0; expert < config.numExperts; expert++) {
                float probability = probabilities[expert];
                int insert = config.numExpertsPerToken - 1;
                if (probability <= topWeights[insert]) {
                    continue;
                }
                while (insert > 0 && probability > topWeights[insert - 1]) {
                    topWeights[insert] = topWeights[insert - 1];
                    topExperts[insert] = topExperts[insert - 1];
                    insert--;
                }
                topWeights[insert] = probability;
                topExperts[insert] = expert;
            }
            if (config.normTopkProb) {
                float topSum = 0.0f;
                for (float weight : topWeights) {
                    topSum += weight;
                }
                for (int top = 0; top < topWeights.length; top++) {
                    topWeights[top] /= topSum;
                }
            }
            int base = row * config.numExpertsPerToken;
            System.arraycopy(topExperts, 0, selectedExperts, base, topExperts.length);
            System.arraycopy(topWeights, 0, selectedWeights, base, topWeights.length);
        }
    }

    private void zero(TensorRef tensor) {
        for (int row = 0; row < tensor.shape().first(); row++) {
            for (int column = 0; column < tensor.shape().last(); column++) {
                tensor.set(0.0f, row, column);
            }
        }
    }
}
