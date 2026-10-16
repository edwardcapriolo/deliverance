package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.tensor2.CompositeOps;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;
import io.teknek.deliverance.tensor2.GemmDit;
import io.teknek.deliverance.tensorlib.TensorPlanAdaptiveSplitTuner;

import java.util.Objects;

/** Projects and normalizes one EmbeddingGemma 2 attention layer's Q/K/V tensors. */
public final class EmbeddingGemma2AttentionProjectionOperation {
    private final int inputWidth;
    private final int queryHeads;
    private final int keyValueHeads;
    private final int headWidth;
    private final float epsilon;

    /** Creates a Q/K/V projection operation for one local or global layer layout. */
    public EmbeddingGemma2AttentionProjectionOperation(int inputWidth, int queryHeads,
                                                       int keyValueHeads, int headWidth, float epsilon) {
        if (inputWidth <= 0 || queryHeads <= 0 || keyValueHeads <= 0 || headWidth <= 0
                || !Float.isFinite(epsilon) || epsilon < 0.0f) {
            throw new IllegalArgumentException("Invalid EmbeddingGemma 2 attention dimensions");
        }
        this.inputWidth = inputWidth;
        this.queryHeads = queryHeads;
        this.keyValueHeads = keyValueHeads;
        this.headWidth = headWidth;
        this.epsilon = epsilon;
    }

    /**
     * Projects all input rows, then applies independent Q/K/V head RMSNorm.
     * Q and K use learned scales; V uses the reference {@code with_scale=false}
     * mode.
     */
    public void apply(TensorRef query, TensorRef key, TensorRef value, TensorRef input,
                       TensorRef queryProjection, TensorRef keyProjection, TensorRef valueProjection,
                       TensorRef queryNorm, TensorRef keyNorm, Lighter lighter) {
        apply(query, key, value, input, queryProjection, keyProjection, valueProjection, queryNorm, keyNorm,
                lighter, null);
    }

    public void apply(TensorRef query, TensorRef key, TensorRef value, TensorRef input,
                       TensorRef queryProjection, TensorRef keyProjection, TensorRef valueProjection,
                       TensorRef queryNorm, TensorRef keyNorm, Lighter lighter,
                       TensorPlanAdaptiveSplitTuner tuner) {
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(queryProjection, "queryProjection");
        Objects.requireNonNull(keyProjection, "keyProjection");
        Objects.requireNonNull(valueProjection, "valueProjection");
        Objects.requireNonNull(lighter, "lighter");
        int queryWidth = queryHeads * headWidth;
        int keyValueWidth = keyValueHeads * headWidth;
        if (input.dims() != 2 || input.shape().last() != inputWidth
                || query.shape().last() != queryWidth || key.shape().last() != keyValueWidth
                || value.shape().last() != keyValueWidth
                || query.shape().first() != input.shape().first()
                || key.shape().first() != input.shape().first()
                || value.shape().first() != input.shape().first()) {
            throw new IllegalArgumentException("Invalid EmbeddingGemma 2 attention tensor shapes");
        }
        GemmDit.dotProductRows(lighter, tuner, "embeddinggemma2.attention.query", query, input, queryProjection,
                0, 0, inputWidth, 0, queryWidth, 0);
        GemmDit.dotProductRows(lighter, tuner, "embeddinggemma2.attention.key", key, input, keyProjection,
                0, 0, inputWidth, 0, keyValueWidth, 0);
        GemmDit.dotProductRows(lighter, tuner, "embeddinggemma2.attention.value", value, input, valueProjection,
                0, 0, inputWidth, 0, keyValueWidth, 0);
        new CompositeOps(lighter).groupedRmsNormInPlace(query, queryHeads, headWidth, epsilon, queryNorm);
        new CompositeOps(lighter).groupedRmsNormInPlace(key, keyValueHeads, headWidth, epsilon, keyNorm);
        new CompositeOps(lighter).groupedRmsNormInPlace(value, keyValueHeads, headWidth, epsilon, null);
    }
}
