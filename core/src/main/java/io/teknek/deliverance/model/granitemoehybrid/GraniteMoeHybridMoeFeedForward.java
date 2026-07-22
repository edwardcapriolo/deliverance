package io.teknek.deliverance.model.granitemoehybrid;

import com.codahale.metrics.Timer;
import io.teknek.deliverance.generator.FeedForward;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import net.jafama.FastMath;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** GraniteMoeHybrid feed-forward branch: sparse MoE plus the always-present shared MLP. */
public class GraniteMoeHybridMoeFeedForward implements FeedForward {

    private static final String METRIC_FORWARD = "granitemoehybrid.moe.forward";
    private static final String METRIC_ROUTE = "granitemoehybrid.moe.route";
    private static final String METRIC_EXPERT = "granitemoehybrid.moe.expert";

    private final AbstractModel model;
    private final GraniteMoeHybridConfig config;
    private final FeedForward sharedMlp;
    private final AbstractTensor routerWeights;
    private final AbstractTensor expertInputWeights;
    private final AbstractTensor expertOutputWeights;
    private final ConfigurableTensorProvider tensorProvider;

    public GraniteMoeHybridMoeFeedForward(AbstractModel model, GraniteMoeHybridConfig config, FeedForward sharedMlp,
            AbstractTensor routerWeights, AbstractTensor expertInputWeights, AbstractTensor expertOutputWeights,
            ConfigurableTensorProvider tensorProvider) {
        this.model = model;
        this.config = config;
        this.sharedMlp = sharedMlp;
        this.routerWeights = routerWeights;
        this.expertInputWeights = expertInputWeights;
        this.expertOutputWeights = expertOutputWeights;
        this.tensorProvider = tensorProvider;
        tensorProvider.get().registerModelTensor(routerWeights);
        tensorProvider.get().registerModelTensor(expertInputWeights);
        tensorProvider.get().registerModelTensor(expertOutputWeights);
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), METRIC_FORWARD).time();
             AbstractTensor moe = forwardMoe(input, tensorReducer)) {
            AbstractTensor shared = sharedMlp.forward(input, tensorReducer);
            tensorProvider.get().accumulate(shared, moe, 0, config.embeddingLength);
            return shared;
        }
    }

    private AbstractTensor forwardMoe(AbstractTensor input, Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        int tokens = input.shape().first();
        AbstractTensor output = model.makeTensor(tokens, config.embeddingLength);
        int[] selectedExperts = new int[tokens * config.numExpertsPerToken];
        float[] selectedWeights = new float[tokens * config.numExpertsPerToken];
        route(input, selectedExperts, selectedWeights);
        int[] expertCounts = countSelectedExperts(tokens, selectedExperts);
        for (int expert = 0; expert < config.numLocalExperts; expert++) {
            int tokenCount = expertCounts[expert];
            if (tokenCount == 0) {
                continue;
            }
            try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), METRIC_EXPERT).time()) {
                forwardExpert(input, output, tensorReducer, selectedExperts, selectedWeights, expert, tokenCount);
            }
        }
        return output;
    }

    private void route(AbstractTensor input, int[] selectedExperts, float[] selectedWeights) {
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), METRIC_ROUTE).time();
             AbstractTensor logits = model.makeTensor(input.shape().first(), config.numLocalExperts)) {
            tensorProvider.get().batchDotProduct(logits, input, routerWeights,
                    0, 0, config.embeddingLength, 0, 0, config.numLocalExperts);
            int[] rowExperts = new int[config.numExpertsPerToken];
            float[] rowWeights = new float[config.numExpertsPerToken];
            for (int token = 0; token < input.shape().first(); token++) {
                topKThenSoftmax(logits, token, rowExperts, rowWeights);
                int offset = token * config.numExpertsPerToken;
                System.arraycopy(rowExperts, 0, selectedExperts, offset, config.numExpertsPerToken);
                System.arraycopy(rowWeights, 0, selectedWeights, offset, config.numExpertsPerToken);
            }
        }
    }

    private void topKThenSoftmax(AbstractTensor logits, int token, int[] rowExperts, float[] rowWeights) {
        Arrays.fill(rowExperts, -1);
        Arrays.fill(rowWeights, Float.NEGATIVE_INFINITY);
        for (int expert = 0; expert < config.numLocalExperts; expert++) {
            float logit = logits.get(token, expert);
            if (logit <= rowWeights[rowWeights.length - 1]) {
                continue;
            }
            int insert = rowWeights.length - 1;
            while (insert > 0 && logit > rowWeights[insert - 1]) {
                rowWeights[insert] = rowWeights[insert - 1];
                rowExperts[insert] = rowExperts[insert - 1];
                insert--;
            }
            rowWeights[insert] = logit;
            rowExperts[insert] = expert;
        }

        float max = Float.NEGATIVE_INFINITY;
        for (float weight : rowWeights) {
            max = Math.max(max, weight);
        }
        double sum = 0.0d;
        for (int i = 0; i < rowWeights.length; i++) {
            rowWeights[i] = (float) FastMath.exp(rowWeights[i] - max);
            sum += rowWeights[i];
        }
        for (int i = 0; i < rowWeights.length; i++) {
            rowWeights[i] /= (float) sum;
        }
    }

    private int[] countSelectedExperts(int tokens, int[] selectedExperts) {
        int[] counts = new int[config.numLocalExperts];
        for (int token = 0; token < tokens; token++) {
            int offset = token * config.numExpertsPerToken;
            for (int top = 0; top < config.numExpertsPerToken; top++) {
                counts[selectedExperts[offset + top]]++;
            }
        }
        return counts;
    }

    private void forwardExpert(AbstractTensor input, AbstractTensor output,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer, int[] selectedExperts, float[] selectedWeights,
            int expert, int tokenCount) {
        try (AbstractTensor expertInput = model.getTensorAllocator().getDirty(input.dType(),
                TensorShape.of(tokenCount, config.embeddingLength));
             AbstractTensor inputProjection = model.makeTensor(tokenCount, config.hiddenLength * 2);
             AbstractTensor hidden = model.makeTensor(tokenCount, config.hiddenLength);
             AbstractTensor down = model.makeTensor(tokenCount, config.embeddingLength);
             AbstractTensor inputWeights = expertInputWeights.slice(expert);
             AbstractTensor outputWeights = expertOutputWeights.slice(expert)) {
            int[] sourceRows = new int[tokenCount];
            float[] routeWeights = new float[tokenCount];
            copySelectedRows(input, selectedExperts, selectedWeights, expert, expertInput, sourceRows, routeWeights);
            tensorProvider.get().batchDotProduct(inputProjection, expertInput, inputWeights,
                    0, 0, config.embeddingLength, 0, 0, config.hiddenLength * 2);
            applyActivationGate(inputProjection, hidden);
            tensorReducer.ifPresent(func -> func.accept(List.of(hidden)));
            try (AbstractTensor hiddenQ = model.maybeQuantize(hidden)) {
                tensorProvider.get().batchDotProduct(down, hiddenQ, outputWeights,
                        0, 0, config.hiddenLength, 0, 0, config.embeddingLength);
            }
            scatterAdd(output, down, sourceRows, routeWeights);
        }
    }

    private void copySelectedRows(AbstractTensor input, int[] selectedExperts, float[] selectedWeights, int expert,
            AbstractTensor expertInput, int[] sourceRows, float[] routeWeights) {
        int outRow = 0;
        for (int token = 0; token < input.shape().first(); token++) {
            int offset = token * config.numExpertsPerToken;
            for (int top = 0; top < config.numExpertsPerToken; top++) {
                int selectedIndex = offset + top;
                if (selectedExperts[selectedIndex] == expert) {
                    expertInput.copyFrom(input, input.getOffset(token, 0), expertInput.getOffset(outRow, 0),
                            config.embeddingLength);
                    sourceRows[outRow] = token;
                    routeWeights[outRow] = selectedWeights[selectedIndex];
                    outRow++;
                }
            }
        }
    }

    private void applyActivationGate(AbstractTensor inputProjection, AbstractTensor hidden) {
        for (int row = 0; row < inputProjection.shape().first(); row++) {
            for (int col = 0; col < config.hiddenLength; col++) {
                float gate = inputProjection.get(row, col);
                float up = inputProjection.get(row, config.hiddenLength + col);
                hidden.set(ActivationFunction.eval(config.activationFunction, gate) * up, row, col);
            }
        }
    }

    private void scatterAdd(AbstractTensor output, AbstractTensor down, int[] sourceRows, float[] routeWeights) {
        for (int row = 0; row < sourceRows.length; row++) {
            int outputRow = sourceRows[row];
            float weight = routeWeights[row];
            for (int col = 0; col < config.embeddingLength; col++) {
                output.set(output.get(outputRow, col) + down.get(row, col) * weight, outputRow, col);
            }
        }
    }
}
