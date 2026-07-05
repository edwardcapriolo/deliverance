package io.teknek.deliverance.model.qwen3;

import com.codahale.metrics.Timer;
import io.teknek.deliverance.generator.FeedForward;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import net.jafama.FastMath;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/** Qwen3 MoE sparse MLP, matching HF {@code Qwen3MoeSparseMoeBlock}. */
public class Qwen3MoeFeedForward implements FeedForward {
    static final ExecutionStrategy SCALAR_EXECUTION = new ExecutionStrategy(true, true, false, false);
    static final ExecutionStrategy PROVIDER_EXECUTION = new ExecutionStrategy(false, false, true, true);
    static final String METRIC_FORWARD = "qwen3moefeedforward.forward";
    static final String METRIC_ROUTE = "qwen3moefeedforward.route";
    static final String METRIC_EXPERT_FORWARD = "qwen3moefeedforward.expert_forward";
    static final String METRIC_EXPERT_GATE_UP_PROJECTION = "qwen3moefeedforward.expert_gate_up_projection";
    static final String METRIC_EXPERT_ACTIVATION_MULTIPLY = "qwen3moefeedforward.expert_activation_multiply";
    static final String METRIC_EXPERT_DOWN_PROJECTION = "qwen3moefeedforward.expert_down_projection";
    static final String METRIC_EXPERT_SCALE_ACCUMULATE = "qwen3moefeedforward.expert_scale_accumulate";
    static final String METRIC_SELECTED_PREFIX = "qwen3moefeedforward.selected_";
    static final String COUNTER_ROUTE_SCALAR = "qwen3moefeedforward.route_scalar";
    static final String COUNTER_ROUTE_PROVIDER = "qwen3moefeedforward.route_provider";
    private final AbstractModel model;
    private final Qwen3MoeConfig config;
    private final AbstractTensor routerWeights;
    private final AbstractTensor[] expertGateWeights;
    private final AbstractTensor[] expertUpWeights;
    private final AbstractTensor[] expertDownWeights;
    private final ConfigurableTensorProvider configurableTensorProvider;
    private final ExecutionStrategy executionStrategy;

    public Qwen3MoeFeedForward(AbstractModel model, Qwen3MoeConfig config, AbstractTensor routerWeights,
            AbstractTensor[] expertGateWeights, AbstractTensor[] expertUpWeights, AbstractTensor[] expertDownWeights,
            ConfigurableTensorProvider configurableTensorProvider) {
        this(model, config, routerWeights, expertGateWeights, expertUpWeights, expertDownWeights,
                configurableTensorProvider, SCALAR_EXECUTION);
    }

    Qwen3MoeFeedForward(AbstractModel model, Qwen3MoeConfig config, AbstractTensor routerWeights,
            AbstractTensor[] expertGateWeights, AbstractTensor[] expertUpWeights, AbstractTensor[] expertDownWeights,
            ConfigurableTensorProvider configurableTensorProvider, ExecutionStrategy executionStrategy) {
        this.model = model;
        this.config = config;
        this.routerWeights = routerWeights;
        this.expertGateWeights = expertGateWeights;
        this.expertUpWeights = expertUpWeights;
        this.expertDownWeights = expertDownWeights;
        this.configurableTensorProvider = configurableTensorProvider;
        this.executionStrategy = executionStrategy;
        configurableTensorProvider.get().registerModelTensor(routerWeights);
        for (int i = 0; i < config.numExperts; i++) {
            configurableTensorProvider.get().registerModelTensor(expertGateWeights[i]);
            configurableTensorProvider.get().registerModelTensor(expertUpWeights[i]);
            configurableTensorProvider.get().registerModelTensor(expertDownWeights[i]);
        }
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        try (Timer.Context ignored = InferenceProfiler.timer(model.getMetricRegistry(), METRIC_FORWARD).time()) {
            if (executionStrategy.providerExpertProjection()) {
                return forwardProvider(input, tensorReducer);
            }
            int batchSize = input.shape().first();
            AbstractTensor output = model.makeTensor(batchSize, config.embeddingLength);
            int[] selectedExperts = new int[config.numExpertsPerToken];
            float[] selectedWeights = new float[config.numExpertsPerToken];
            for (int batch = 0; batch < batchSize; batch++) {
                try (Timer.Context ignoredRoute = InferenceProfiler.timer(model.getMetricRegistry(), METRIC_ROUTE).time()) {
                    route(input, batch, selectedExperts, selectedWeights);
                }
                for (int top = 0; top < config.numExpertsPerToken; top++) {
                    InferenceProfiler.counter(model.getMetricRegistry(), METRIC_SELECTED_PREFIX + selectedExperts[top]).inc();
                    try (Timer.Context ignoredExpert = InferenceProfiler.timer(model.getMetricRegistry(), METRIC_EXPERT_FORWARD).time()) {
                        accumulateExpert(input, output, batch, selectedExperts[top], selectedWeights[top]);
                    }
                }
            }
            return output;
        }
    }

    private AbstractTensor forwardProvider(AbstractTensor input, Optional<Consumer<List<AbstractTensor>>> tensorReducer) {
        int batchSize = input.shape().first();
        AbstractTensor output = model.makeTensor(batchSize, config.embeddingLength);
        int[] selectedExperts = new int[batchSize * config.numExpertsPerToken];
        float[] selectedWeights = new float[batchSize * config.numExpertsPerToken];
        routeBatch(input, selectedExperts, selectedWeights);
        int[] expertCounts = countSelectedExperts(batchSize, selectedExperts);
        for (int expert = 0; expert < config.numExperts; expert++) {
            if (expertCounts[expert] == 0) {
                continue;
            }
            InferenceProfiler.counter(model.getMetricRegistry(), METRIC_SELECTED_PREFIX + expert).inc(expertCounts[expert]);
            try (Timer.Context ignoredExpert = InferenceProfiler.timer(model.getMetricRegistry(), METRIC_EXPERT_FORWARD).time()) {
                forwardProviderExpert(input, output, tensorReducer, selectedExperts, selectedWeights, expert, expertCounts[expert]);
            }
        }
        return output;
    }

    private int[] countSelectedExperts(int batchSize, int[] selectedExperts) {
        int[] counts = new int[config.numExperts];
        for (int batch = 0; batch < batchSize; batch++) {
            int base = batch * config.numExpertsPerToken;
            for (int top = 0; top < config.numExpertsPerToken; top++) {
                counts[selectedExperts[base + top]]++;
            }
        }
        return counts;
    }

    private void forwardProviderExpert(AbstractTensor input, AbstractTensor output,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer, int[] selectedExperts, float[] selectedWeights,
            int expert, int tokenCount) {
        try (AbstractTensor expertInput = model.getTensorAllocator().getDirty(input.dType(), TensorShape.of(tokenCount, config.embeddingLength));
             AbstractTensor gate = model.makeTensor(tokenCount, config.moeIntermediateSize);
             AbstractTensor up = model.makeTensor(tokenCount, config.moeIntermediateSize);
             AbstractTensor hidden = model.makeTensor(tokenCount, config.moeIntermediateSize);
             AbstractTensor down = model.makeTensor(tokenCount, config.embeddingLength)) {
            int[] sourceRows = new int[tokenCount];
            float[] routeWeights = new float[tokenCount];
            copySelectedRows(input, selectedExperts, selectedWeights, expert, expertInput, sourceRows, routeWeights);
            try (Timer.Context ignoredGateUp = InferenceProfiler.timer(model.getMetricRegistry(), METRIC_EXPERT_GATE_UP_PROJECTION).time()) {
                configurableTensorProvider.get().batchDotProduct(gate, expertInput, expertGateWeights[expert],
                        0, 0, config.embeddingLength, 0, 0, config.moeIntermediateSize);
                configurableTensorProvider.get().batchDotProduct(up, expertInput, expertUpWeights[expert],
                        0, 0, config.embeddingLength, 0, 0, config.moeIntermediateSize);
            }
            try (Timer.Context ignoredActivation = InferenceProfiler.timer(model.getMetricRegistry(), METRIC_EXPERT_ACTIVATION_MULTIPLY).time()) {
                for (int row = 0; row < tokenCount; row++) {
                    for (int col = 0; col < config.moeIntermediateSize; col++) {
                        hidden.set(ActivationFunction.eval(config.activationFunction, gate.get(row, col)) * up.get(row, col), row, col);
                    }
                }
            }
            tensorReducer.ifPresent(func -> func.accept(List.of(hidden)));
            try (AbstractTensor hiddenQ = model.maybeQuantize(hidden);
                 Timer.Context ignoredDown = InferenceProfiler.timer(model.getMetricRegistry(), METRIC_EXPERT_DOWN_PROJECTION).time()) {
                configurableTensorProvider.get().batchDotProduct(down, hiddenQ, expertDownWeights[expert],
                        0, 0, config.moeIntermediateSize, 0, 0, config.embeddingLength);
            }
            try (Timer.Context ignoredScale = InferenceProfiler.timer(model.getMetricRegistry(), METRIC_EXPERT_SCALE_ACCUMULATE).time()) {
                scatterAdd(output, down, sourceRows, routeWeights);
            }
        }
    }

    private void copySelectedRows(AbstractTensor input, int[] selectedExperts, float[] selectedWeights, int expert,
            AbstractTensor expertInput, int[] sourceRows, float[] routeWeights) {
        int outRow = 0;
        for (int batch = 0; batch < input.shape().first(); batch++) {
            int base = batch * config.numExpertsPerToken;
            for (int top = 0; top < config.numExpertsPerToken; top++) {
                int selectedIndex = base + top;
                if (selectedExperts[selectedIndex] == expert) {
                    expertInput.copyFrom(input, input.getOffset(batch, 0), expertInput.getOffset(outRow, 0), config.embeddingLength);
                    sourceRows[outRow] = batch;
                    routeWeights[outRow] = selectedWeights[selectedIndex];
                    outRow++;
                }
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

    ExecutionStrategy executionStrategy() {
        return executionStrategy;
    }

    record ExecutionStrategy(
            boolean tokenByTokenRouting,
            boolean scalarExpertAccumulation,
            boolean providerRouterProjection,
            boolean providerExpertProjection
    ) {
    }

    void route(AbstractTensor input, int batch, int[] selectedExperts, float[] selectedWeights) {
        if (executionStrategy.providerRouterProjection()) {
            routeWithProviderProjection(input, batch, selectedExperts, selectedWeights);
        } else {
            routeScalar(input, batch, selectedExperts, selectedWeights);
        }
    }

    void routeBatch(AbstractTensor input, int[] selectedExperts, float[] selectedWeights) {
        int[] rowExperts = new int[config.numExpertsPerToken];
        float[] rowWeights = new float[config.numExpertsPerToken];
        if (!executionStrategy.providerRouterProjection()) {
            for (int batch = 0; batch < input.shape().first(); batch++) {
                routeScalar(input, batch, rowExperts, rowWeights);
                copyRouteRow(selectedExperts, selectedWeights, batch, rowExperts, rowWeights);
            }
            return;
        }
        InferenceProfiler.counter(model.getMetricRegistry(), COUNTER_ROUTE_PROVIDER).inc(input.shape().first());
        try (AbstractTensor logits = model.makeTensor(input.shape().first(), config.numExperts)) {
            configurableTensorProvider.get().batchDotProduct(logits, input, routerWeights,
                    0, 0, config.embeddingLength, 0, 0, config.numExperts);
            for (int batch = 0; batch < input.shape().first(); batch++) {
                routeFromLogits(logits, batch, rowExperts, rowWeights);
                copyRouteRow(selectedExperts, selectedWeights, batch, rowExperts, rowWeights);
            }
        }
    }

    private void copyRouteRow(int[] selectedExperts, float[] selectedWeights, int batch,
            int[] rowExperts, float[] rowWeights) {
        int offset = batch * config.numExpertsPerToken;
        System.arraycopy(rowExperts, 0, selectedExperts, offset, config.numExpertsPerToken);
        System.arraycopy(rowWeights, 0, selectedWeights, offset, config.numExpertsPerToken);
    }

    void routeScalar(AbstractTensor input, int batch, int[] selectedExperts, float[] selectedWeights) {
        InferenceProfiler.counter(model.getMetricRegistry(), COUNTER_ROUTE_SCALAR).inc();
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
        softmaxAndTopK(probabilities, max, selectedExperts, selectedWeights);
    }

    void routeWithProviderProjection(AbstractTensor input, int batch, int[] selectedExperts, float[] selectedWeights) {
        InferenceProfiler.counter(model.getMetricRegistry(), COUNTER_ROUTE_PROVIDER).inc();
        float[] probabilities = new float[config.numExperts];
        float max = Float.NEGATIVE_INFINITY;
        try (AbstractTensor logits = model.makeTensor(1, config.numExperts);
             AbstractTensor inputRow = input.slice(batch)) {
            configurableTensorProvider.get().batchDotProduct(logits, inputRow, routerWeights,
                    0, 0, config.embeddingLength, 0, 0, config.numExperts);
            for (int expert = 0; expert < config.numExperts; expert++) {
                float score = logits.get(0, expert);
                probabilities[expert] = score;
                max = Math.max(max, score);
            }
        }
        softmaxAndTopK(probabilities, max, selectedExperts, selectedWeights);
    }

    void routeFromLogits(AbstractTensor logits, int batch, int[] selectedExperts, float[] selectedWeights) {
        float[] probabilities = new float[config.numExperts];
        float max = Float.NEGATIVE_INFINITY;
        for (int expert = 0; expert < config.numExperts; expert++) {
            float score = logits.get(batch, expert);
            probabilities[expert] = score;
            max = Math.max(max, score);
        }
        softmaxAndTopK(probabilities, max, selectedExperts, selectedWeights);
    }

    private void softmaxAndTopK(float[] probabilities, float max, int[] selectedExperts, float[] selectedWeights) {
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
            normalizeTopKWeights(selectedWeights);
        }
    }

    private void normalizeTopKWeights(float[] selectedWeights) {
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

    void topK(float[] probabilities, int[] selectedExperts, float[] selectedWeights) {
        java.util.Arrays.fill(selectedExperts, -1);
        java.util.Arrays.fill(selectedWeights, Float.NEGATIVE_INFINITY);
        for (int expert = 0; expert < probabilities.length; expert++) {
            float probability = probabilities[expert];
            if (probability <= selectedWeights[selectedWeights.length - 1]) {
                continue;
            }
            int insert = selectedWeights.length - 1;
            while (insert > 0 && probability > selectedWeights[insert - 1]) {
                selectedWeights[insert] = selectedWeights[insert - 1];
                selectedExperts[insert] = selectedExperts[insert - 1];
                insert--;
            }
            selectedWeights[insert] = probability;
            selectedExperts[insert] = expert;
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
