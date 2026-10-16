package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.TensorShape;
import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor2.GemmDit;
import io.teknek.deliverance.tensorlib.TensorPlanAdaptiveSplitTuner;

import java.util.Objects;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

/** Executes one complete bidirectional EmbeddingGemma 2 encoder layer. */
public final class EmbeddingGemma2EncoderLayerOperation {
    private final int hiddenSize;
    private final int intermediateSize;
    private final int perLayerInputWidth;
    private final int queryHeads;
    private final int keyValueHeads;
    private final int headWidth;
    private final float epsilon;
    private final String layerType;
    private final int slidingWindow;
    private final double ropeTheta;
    private final float attentionScale;
    private final boolean roundBf16;
    private final EmbeddingGemma2RmsNormOperation rmsNorm;
    private final EmbeddingGemma2AttentionProjectionOperation attentionProjection;
    private final EmbeddingGemma2AttentionScoreOperation attentionScores;
    private final EmbeddingGemma2AttentionSoftmaxOperation attentionSoftmax;
    private final EmbeddingGemma2AttentionValueOperation attentionValues;
    private final EmbeddingGemma2MlpOperation mlp;
    private final EmbeddingGemma2PleBlockOperation ple;
    private final EmbeddingGemma2RotaryEmbeddingOperation queryRotary;
    private final EmbeddingGemma2RotaryEmbeddingOperation keyRotary;
    private final ForkJoinPool pool;

    /** Creates a layer with the configured local/global attention contract. */
    public EmbeddingGemma2EncoderLayerOperation(int hiddenSize, int intermediateSize, int queryHeads,
            int keyValueHeads, int headWidth, int perLayerInputWidth, float epsilon, String layerType,
            int slidingWindow, double ropeTheta, boolean roundBf16, int maxPositions, ForkJoinPool pool) {
        if (hiddenSize <= 0 || intermediateSize <= 0 || queryHeads <= 0 || keyValueHeads <= 0
                || headWidth <= 0 || perLayerInputWidth <= 0 || queryHeads % keyValueHeads != 0
                || !Float.isFinite(epsilon)
                || epsilon < 0.0f) {
            throw new IllegalArgumentException("Invalid EmbeddingGemma 2 encoder layer configuration");
        }
        this.hiddenSize = hiddenSize;
        this.intermediateSize = intermediateSize;
        this.perLayerInputWidth = perLayerInputWidth;
        this.queryHeads = queryHeads;
        this.keyValueHeads = keyValueHeads;
        this.headWidth = headWidth;
        this.epsilon = epsilon;
        this.layerType = Objects.requireNonNull(layerType, "layerType");
        this.slidingWindow = slidingWindow;
        if (!Double.isFinite(ropeTheta) || ropeTheta <= 0.0) {
            throw new IllegalArgumentException("RoPE theta must be positive and finite");
        }
        this.ropeTheta = ropeTheta;
        this.roundBf16 = roundBf16;
        this.attentionScale = 1.0f;
        this.rmsNorm = new EmbeddingGemma2RmsNormOperation(epsilon);
        this.attentionProjection = new EmbeddingGemma2AttentionProjectionOperation(hiddenSize, queryHeads,
                keyValueHeads, headWidth, epsilon);
        this.attentionScores = new EmbeddingGemma2AttentionScoreOperation(queryHeads, keyValueHeads,
                attentionScale);
        this.attentionSoftmax = new EmbeddingGemma2AttentionSoftmaxOperation();
        this.attentionValues = new EmbeddingGemma2AttentionValueOperation(queryHeads, keyValueHeads, headWidth);
        this.mlp = new EmbeddingGemma2MlpOperation(hiddenSize, intermediateSize, roundBf16);
        this.ple = new EmbeddingGemma2PleBlockOperation(hiddenSize, perLayerInputWidth, epsilon);
        this.pool = Objects.requireNonNull(pool, "pool");
        this.queryRotary = new EmbeddingGemma2RotaryEmbeddingOperation(headWidth, queryHeads, ropeTheta,
                maxPositions);
        this.keyRotary = new EmbeddingGemma2RotaryEmbeddingOperation(headWidth, keyValueHeads, ropeTheta,
                maxPositions);
    }

    /** Runs attention, MLP, and PLE residual updates over all token rows. */
    public void apply(TensorRef output, TensorRef input, TensorRef perLayerInput,
            TensorRef inputNormWeight, TensorRef queryWeight, TensorRef keyWeight, TensorRef valueWeight,
            TensorRef outputWeight, TensorRef queryNormWeight, TensorRef keyNormWeight,
            TensorRef postAttentionNormWeight, TensorRef preFeedForwardNormWeight,
            TensorRef gateWeight, TensorRef upWeight, TensorRef downWeight,
            TensorRef postFeedForwardNormWeight, TensorRef pleGateWeight, TensorRef pleProjectionWeight,
            TensorRef pleNormWeight, float layerScalar, int[] attentionMask, Lighter lighter) {
        apply(output, input, perLayerInput, inputNormWeight, queryWeight, keyWeight, valueWeight, outputWeight,
                queryNormWeight, keyNormWeight, postAttentionNormWeight, preFeedForwardNormWeight, gateWeight,
                upWeight, downWeight, postFeedForwardNormWeight, pleGateWeight, pleProjectionWeight, pleNormWeight,
                layerScalar, attentionMask, lighter, null);
    }

    public void apply(TensorRef output, TensorRef input, TensorRef perLayerInput,
            TensorRef inputNormWeight, TensorRef queryWeight, TensorRef keyWeight, TensorRef valueWeight,
            TensorRef outputWeight, TensorRef queryNormWeight, TensorRef keyNormWeight,
            TensorRef postAttentionNormWeight, TensorRef preFeedForwardNormWeight,
            TensorRef gateWeight, TensorRef upWeight, TensorRef downWeight,
            TensorRef postFeedForwardNormWeight, TensorRef pleGateWeight, TensorRef pleProjectionWeight,
            TensorRef pleNormWeight, float layerScalar, int[] attentionMask, Lighter lighter,
            TensorPlanAdaptiveSplitTuner tuner) {
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(perLayerInput, "perLayerInput");
        Objects.requireNonNull(lighter, "lighter");
        validateShape(output, input, perLayerInput);
        try (TensorRef normalized = allocateRows(lighter, input);
             TensorRef query = allocateRows(lighter, input, queryHeads * headWidth);
             TensorRef key = allocateRows(lighter, input, keyValueHeads * headWidth);
             TensorRef value = allocateRows(lighter, input, keyValueHeads * headWidth);
             TensorRef attentionOutput = allocateRows(lighter, input, queryHeads * headWidth);
             TensorRef projectedAttention = allocateRows(lighter, input);
             TensorRef normalizedAttention = allocateRows(lighter, input);
             TensorRef feedForwardInput = allocateRows(lighter, input);
             TensorRef feedForward = allocateRows(lighter, input);
             TensorRef projectedFeedForward = allocateRows(lighter, input)) {
            rmsNorm.apply(normalized, input, inputNormWeight, lighter);
            attentionProjection.apply(query, key, value, normalized, queryWeight, keyWeight, valueWeight,
                    queryNormWeight, keyNormWeight, lighter, tuner);
            roundBoundary(query, lighter);
            roundBoundary(key, lighter);
            roundBoundary(value, lighter);
            queryRotary.apply(query, 0, lighter);
            keyRotary.apply(key, 0, lighter);
            roundBoundary(query, lighter);
            roundBoundary(key, lighter);
            TensorRef[] queryHeadsRefs = rowsByHead(query, queryHeads, headWidth, lighter);
            TensorRef[] keyHeadsRefs = rowsByHead(key, keyValueHeads, headWidth, lighter);
            TensorRef[] scoreRefs = new TensorRef[queryHeads];
            try {
                for (int head = 0; head < queryHeads; head++) {
                    scoreRefs[head] = lighter.allocate(DType.F32,
                            TensorShape.of(input.shape().first(), input.shape().first()));
                }
                lighter.clear(attentionOutput);
                List<ForkJoinTask<?>> headTasks = new ArrayList<>(queryHeads);
                for (int head = 0; head < queryHeads; head++) {
                    int taskHead = head;
                    headTasks.add(pool.submit(() -> {
                        attentionScores.applyHead(queryHeadsRefs[taskHead],
                                keyHeadsRefs[taskHead / (queryHeads / keyValueHeads)], scoreRefs[taskHead],
                                layerType, slidingWindow, attentionMask, lighter);
                        attentionSoftmax.apply(new TensorRef[] {scoreRefs[taskHead]}, lighter);
                        attentionValues.applyHead(scoreRefs[taskHead], value, attentionOutput, taskHead, lighter);
                    }));
                }
                for (ForkJoinTask<?> headTask : headTasks) {
                    headTask.join();
                }
                roundBoundary(attentionOutput, lighter);
            } finally {
                closeAll(queryHeadsRefs);
                closeAll(keyHeadsRefs);
                closeAll(scoreRefs);
            }
            GemmDit.dotProductRows(lighter, tuner, "embeddinggemma2.attention.output", projectedAttention,
                    attentionOutput, outputWeight, 0, 0, queryHeads * headWidth, 0, hiddenSize, 0);
            roundBoundary(projectedAttention, lighter);
            rmsNorm.apply(normalizedAttention, projectedAttention, postAttentionNormWeight, lighter);
            roundBoundary(normalizedAttention, lighter);
            new CompositeOps(lighter).addResidual(normalizedAttention, input, 1.0f);
            roundBoundary(normalizedAttention, lighter);
            rmsNorm.apply(feedForwardInput, normalizedAttention, preFeedForwardNormWeight, lighter);
            roundBoundary(feedForwardInput, lighter);
            mlp.apply(feedForward, feedForwardInput, gateWeight, upWeight, downWeight, lighter, tuner);
            rmsNorm.apply(projectedFeedForward, feedForward, postFeedForwardNormWeight, lighter);
            roundBoundary(projectedFeedForward, lighter);
            new CompositeOps(lighter).addResidual(projectedFeedForward, normalizedAttention, 1.0f);
            roundBoundary(projectedFeedForward, lighter);
            ple.apply(output, projectedFeedForward, perLayerInput, pleGateWeight, pleProjectionWeight,
                    pleNormWeight, lighter);
            if (layerScalar != 1.0f) {
                lighter.multiplyAll(layerScalar, output);
            }
        }
    }

    private void roundBoundary(TensorRef target, Lighter lighter) {
        if (!roundBf16) return;
        try (TensorRef bf16 = lighter.reshape(target, DType.BF16,
                io.teknek.deliverance.tensor2.Threading.PROVIDER)) {
            lighter.reshape(bf16, target, io.teknek.deliverance.tensor2.Threading.PROVIDER);
        }
    }

    private TensorRef allocateRows(Lighter lighter, TensorRef input) {
        return allocateRows(lighter, input, hiddenSize);
    }

    private TensorRef allocateRows(Lighter lighter, TensorRef input, int width) {
        return lighter.allocate(DType.F32, TensorShape.of(input.shape().first(), width));
    }

    private TensorRef[] rowsByHead(TensorRef tensor, int heads, int width, Lighter lighter) {
        TensorRef[] result = new TensorRef[heads];
        for (int head = 0; head < heads; head++) {
            result[head] = lighter.allocate(DType.F32, TensorShape.of(tensor.shape().first(), width));
            for (int row = 0; row < tensor.shape().first(); row++) {
                for (int column = 0; column < width; column++) {
                    result[head].set(tensor.get(row, head * width + column), row, column);
                }
            }
        }
        return result;
    }

    private void closeAll(TensorRef[] tensors) {
        for (TensorRef tensor : tensors) {
            if (tensor != null) {
                tensor.close();
            }
        }
    }

    private void validateShape(TensorRef output, TensorRef input, TensorRef perLayerInput) {
        if (input.dims() != 2 || output.dims() != 2 || perLayerInput.dims() != 2
                || !input.shape().equals(output.shape()) || input.shape().first() != perLayerInput.shape().first()
                || input.shape().last() != hiddenSize || perLayerInput.shape().last() != perLayerInputWidth) {
            throw new IllegalArgumentException("Invalid EmbeddingGemma 2 encoder layer tensor shapes");
        }
    }
}
