package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Objects;

/** Applies a numerically stable row-wise softmax to masked attention scores. */
public final class EmbeddingGemma2AttentionSoftmaxOperation {
    /** Mask sentinel written by the score operation. */
    private static final float MASKED_SCORE = -Float.MAX_VALUE;

    /** Applies softmax independently to every query row of every head. */
    public void apply(TensorRef[] scores, Lighter lighter) {
        Objects.requireNonNull(scores, "scores");
        Objects.requireNonNull(lighter, "lighter");
        if (scores.length == 0) {
            throw new IllegalArgumentException("At least one attention score head is required");
        }
        for (TensorRef score : scores) {
            Objects.requireNonNull(score, "score");
            if (score.dims() != 2) {
                throw new IllegalArgumentException("Attention scores must be rank 2");
            }
            for (int row = 0; row < score.shape().first(); row++) {
                softmaxRow(score, row);
            }
        }
    }

    /** Normalizes one row while preserving zero probability for masked entries. */
    private void softmaxRow(TensorRef score, int row) {
        float maximum = -Float.MAX_VALUE;
        boolean hasVisibleKey = false;
        for (int key = 0; key < score.shape().last(); key++) {
            float value = score.get(row, key);
            if (value != MASKED_SCORE) {
                hasVisibleKey = true;
                maximum = Math.max(maximum, value);
            }
        }
        if (!hasVisibleKey) {
            for (int key = 0; key < score.shape().last(); key++) {
                score.set(0.0f, row, key);
            }
            return;
        }

        float sum = 0.0f;
        for (int key = 0; key < score.shape().last(); key++) {
            float value = score.get(row, key);
            float probability = value == MASKED_SCORE ? 0.0f : (float) Math.exp(value - maximum);
            score.set(probability, row, key);
            sum += probability;
        }
        for (int key = 0; key < score.shape().last(); key++) {
            score.set(score.get(row, key) / sum, row, key);
        }
    }
}
