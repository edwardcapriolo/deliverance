package io.teknek.deliverance.model.gemma4;

import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.generator2.FeedForward2;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.tensor2.ActivationMultiplyQuantize;
import io.teknek.deliverance.tensor2.BatchDotProduct;
import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.MultiplyInPlace;
import io.teknek.deliverance.tensor2.MultiplyAccumulate;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * Gemma4 packed-expert feed-forward operation. Matrix work is dispatched through Lighter; the
 * small routing loop only selects expert metadata and never performs tensor element arithmetic.
 */
final class Gemma4MoeFeedForward2 implements FeedForward2 {
    private final AbstractModel model;
    private final Gemma4Config config;
    private final TensorRef router;
    private final TensorRef routerScale;
    private final TensorRef perExpertScale;
    private final TensorRef gateUp;
    private final TensorRef down;
    private final Lighter lighter;
    private final CompositeOps ops;

    Gemma4MoeFeedForward2(AbstractModel model, Gemma4Config config, TensorRef router,
            TensorRef routerScale, TensorRef perExpertScale, TensorRef gateUp, TensorRef down, Lighter lighter) {
        this.model = model;
        this.config = config;
        this.router = router;
        this.routerScale = routerScale;
        this.perExpertScale = perExpertScale;
        this.gateUp = gateUp;
        this.down = down;
        this.lighter = lighter;
        this.ops = new CompositeOps(lighter, model.getMetricRegistry());
    }

    @Override
    public TensorRef forward(TensorRef input, Optional<Consumer<List<TensorRef>>> reducer, ForwardPhase phase) {
        int rows = input.shape().first();
        TensorRef output = model.makeDenseTensorRef(rows, config.embeddingLength);
        TensorRef logits = model.makeDenseTensorRef(rows, config.numExperts);
        TensorRef normalized = model.makeDenseTensorRef(input.shape());
        TensorRef routerScale2 = lighter.allocate(routerScale.dType(),
                io.teknek.deliverance.tensor.TensorShape.of(1, config.embeddingLength));
        try {
            lighter.reshape(input, normalized);
            lighter.copy(routerScale, 0, routerScale2, 0, config.embeddingLength);
            ops.groupedRmsNormInPlace(normalized, 1, config.embeddingLength, config.layerNormEps, routerScale2);
            ops.multiplyInPlace(new MultiplyInPlace((float) Math.pow(config.embeddingLength, -0.5f))
                    .target(normalized).offsetAndLength(0, config.embeddingLength));
            lighter.batchDotProduct(new BatchDotProduct().result(logits).a(normalized).b(router)
                    .aColumnOffset(0).bColumnOffset(0).columnLength(config.embeddingLength)
                    .resultRowOffset(0).bRowOffset(0).rowChunkSize(config.numExperts));
            lighter.clear(output);
            applyMoeBranch(input, output, logits, phase);
            reducer.ifPresent(r -> r.accept(Collections.singletonList(output)));
            return output;
        } catch (RuntimeException | Error e) {
            output.close();
            throw e;
        } finally {
            logits.close();
            normalized.close();
            routerScale2.close();
        }
    }

    @Override
    public TensorRef forward(TensorRef input, Optional<Consumer<List<TensorRef>>> reducer) {
        return forward(input, reducer, ForwardPhase.DECODE);
    }

    private void applyMoeBranch(TensorRef input, TensorRef output, TensorRef logits, ForwardPhase phase) {
        for (int row = 0; row < input.shape().first(); row++) {
            float[] probabilities = computeRouterProbabilities(logits, row);
            int[] experts = selectTopKExperts(probabilities);
            float sum = 0;
            for (int expert : experts) sum += probabilities[expert];
            for (int expert : experts) {
                float weight = probabilities[expert] / sum * scaleAt(perExpertScale, expert);
                applyMoeExpert(input, output, row, expert, weight, phase);
            }
        }
    }

    private static float scaleAt(TensorRef scales, int index) {
        return scales.dims() == 1 ? scales.get(index) : scales.get(0, index);
    }

    private void applyMoeExpert(TensorRef input, TensorRef output, int row, int expert, float weight, ForwardPhase phase) {
        try (TensorRef source = model.makeDenseTensorRef(1, config.embeddingLength);
             TensorRef target = model.makeDenseTensorRef(1, config.embeddingLength);
             TensorRef expertGateUp = gateUp.slice(expert);
             TensorRef expertDown = down.slice(expert);
             TensorRef gate = model.makeDenseTensorRef(1, config.moeIntermediateSize);
             TensorRef up = model.makeDenseTensorRef(1, config.moeIntermediateSize)) {
            lighter.copy(input, input.shape().getOffset(row, 0), source, 0, config.embeddingLength);
            lighter.copy(output, output.shape().getOffset(row, 0), target, 0, config.embeddingLength);
            lighter.dotProductRows(gate, source, expertGateUp, 0, 0, config.embeddingLength, 0,
                    config.moeIntermediateSize, 0);
            lighter.dotProductRows(up, source, expertGateUp, 0, 0, config.embeddingLength,
                    config.moeIntermediateSize, config.moeIntermediateSize, 0);
            ops.silu(gate);
            lighter.multiplyAccumulate(new MultiplyAccumulate(up).into(gate)
                    .offsetAndLength(0, config.moeIntermediateSize));
            try (TensorRef result = model.makeDenseTensorRef(1, config.embeddingLength)) {
                lighter.dotProductRows(result, gate, expertDown, 0, 0, config.moeIntermediateSize, 0,
                        config.embeddingLength, 0);
                if (weight == 1.0f) {
                    lighter.copy(result, 0, target, 0, config.embeddingLength);
                } else {
                    lighter.saxpy(weight, result, target, 0, 0, config.embeddingLength);
                }
                lighter.copy(target, 0, output, output.shape().getOffset(row, 0), config.embeddingLength);
            }
        }
    }

    private float[] computeRouterProbabilities(TensorRef logits, int row) {
        float[] result = new float[config.numExperts];
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < result.length; i++) {
            result[i] = logits.get(row, i);
            max = Math.max(max, result[i]);
        }
        float sum = 0;
        for (int i = 0; i < result.length; i++) sum += result[i] = (float) Math.exp(result[i] - max);
        for (int i = 0; i < result.length; i++) result[i] /= sum;
        return result;
    }

    private int[] selectTopKExperts(float[] probabilities) {
        int[] result = new int[config.topKExperts];
        Arrays.fill(result, -1);
        for (int expert = 0; expert < probabilities.length; expert++) {
            int insert = result.length - 1;
            while (insert > 0 && (result[insert - 1] < 0 || probabilities[expert] > probabilities[result[insert - 1]])) {
                result[insert] = result[insert - 1];
                insert--;
            }
            if (result[insert] < 0 || probabilities[expert] > probabilities[result[insert]]) result[insert] = expert;
        }
        return result;
    }
}
