package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Objects;

/** Aggregates value rows using per-head attention probabilities. */
public final class EmbeddingGemma2AttentionValueOperation {
    private final int queryHeads;
    private final int keyValueHeads;
    private final int headWidth;

    /** Creates a value aggregation operation with grouped-query head mapping. */
    public EmbeddingGemma2AttentionValueOperation(int queryHeads, int keyValueHeads, int headWidth) {
        if (queryHeads <= 0 || keyValueHeads <= 0 || headWidth <= 0 || queryHeads % keyValueHeads != 0) {
            throw new IllegalArgumentException("Invalid EmbeddingGemma 2 attention value configuration");
        }
        this.queryHeads = queryHeads;
        this.keyValueHeads = keyValueHeads;
        this.headWidth = headWidth;
    }

    /** Computes {@code output = attentionProbabilities * values} for every query head. */
    public void apply(TensorRef[] scores, TensorRef values, TensorRef output, Lighter lighter) {
        Objects.requireNonNull(scores, "scores");
        Objects.requireNonNull(values, "values");
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(lighter, "lighter");
        if (scores.length != queryHeads || values.dims() != 2 || output.dims() != 2
                || values.shape().last() != keyValueHeads * headWidth
                || output.shape().last() != queryHeads * headWidth
                || output.shape().first() != values.shape().first()) {
            throw new IllegalArgumentException("Invalid EmbeddingGemma 2 attention value tensor shapes");
        }
        lighter.clear(output);
        for (int head = 0; head < queryHeads; head++) {
            TensorRef score = scores[head];
            if (score == null || score.dims() != 2 || score.shape().first() != output.shape().first()
                    || score.shape().last() != values.shape().first()) {
                throw new IllegalArgumentException("Attention score shape does not match values");
            }
            int valueHead = head / (queryHeads / keyValueHeads);
            int outputOffset = head * headWidth;
            int valueOffset = valueHead * headWidth;
            for (int query = 0; query < output.shape().first(); query++) {
                for (int dimension = 0; dimension < headWidth; dimension++) {
                    float sum = 0.0f;
                    for (int key = 0; key < values.shape().first(); key++) {
                        sum += score.get(query, key) * values.get(key, valueOffset + dimension);
                    }
                    output.set(sum, query, outputOffset + dimension);
                }
            }
        }
    }
}
