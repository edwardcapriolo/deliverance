package io.teknek.deliverance.embedding;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.math.VectorMathUtils;
import io.teknek.deliverance.tensor.AbstractTensor;

/** SentenceTransformers-compatible pooling helpers for flattened [batch, seq, hidden] token embeddings. */
public final class SentenceTransformersPooling {
    public enum Mode {
        CLS,
        MAX,
        MEAN,
        MEAN_SQRT_LEN_TOKENS,
        WEIGHTED_MEAN,
        LAST_TOKEN
    }

    private SentenceTransformersPooling() {
    }

    public static float[][] pool(AbstractTensor tokenEmbeddings, int[] attentionMask, int batchSize, int sequenceLength,
            Mode... modes) {
        Preconditions.checkArgument(tokenEmbeddings.dims() == 2, "tokenEmbeddings must be flattened 2D");
        Preconditions.checkArgument(tokenEmbeddings.shape().first() == (long) batchSize * sequenceLength,
                "tokenEmbeddings rows must equal batchSize * sequenceLength");
        Preconditions.checkArgument(attentionMask.length == batchSize * sequenceLength,
                "attentionMask length must equal batchSize * sequenceLength");
        int hidden = (int) tokenEmbeddings.shape().last();
        float[][] output = new float[batchSize][hidden * modes.length];
        for (int batch = 0; batch < batchSize; batch++) {
            int outputOffset = 0;
            for (Mode mode : modes) {
                float[] pooled = switch (mode) {
                    case CLS -> cls(tokenEmbeddings, attentionMask, batch, sequenceLength, hidden);
                    case MAX -> max(tokenEmbeddings, attentionMask, batch, sequenceLength, hidden);
                    case MEAN -> mean(tokenEmbeddings, attentionMask, batch, sequenceLength, hidden, false);
                    case MEAN_SQRT_LEN_TOKENS -> mean(tokenEmbeddings, attentionMask, batch, sequenceLength, hidden, true);
                    case WEIGHTED_MEAN -> weightedMean(tokenEmbeddings, attentionMask, batch, sequenceLength, hidden);
                    case LAST_TOKEN -> lastToken(tokenEmbeddings, attentionMask, batch, sequenceLength, hidden);
                };
                System.arraycopy(pooled, 0, output[batch], outputOffset, hidden);
                outputOffset += hidden;
            }
        }
        return output;
    }

    public static void normalize(float[] embedding) {
        VectorMathUtils.l2normalize(embedding);
    }

    private static float[] cls(AbstractTensor tokenEmbeddings, int[] attentionMask, int batch, int sequenceLength,
            int hidden) {
        int first = batch * sequenceLength;
        for (int token = 0; token < sequenceLength; token++) {
            int row = batch * sequenceLength + token;
            if (attentionMask[row] != 0) {
                return row(tokenEmbeddings, row, hidden);
            }
        }
        return new float[hidden];
    }

    private static float[] max(AbstractTensor tokenEmbeddings, int[] attentionMask, int batch, int sequenceLength,
            int hidden) {
        float[] output = new float[hidden];
        boolean seen = false;
        for (int i = 0; i < hidden; i++) {
            output[i] = -1.0e9f;
        }
        for (int token = 0; token < sequenceLength; token++) {
            int row = batch * sequenceLength + token;
            if (attentionMask[row] == 0) {
                continue;
            }
            seen = true;
            for (int col = 0; col < hidden; col++) {
                output[col] = Math.max(output[col], tokenEmbeddings.get(row, col));
            }
        }
        return seen ? output : new float[hidden];
    }

    private static float[] mean(AbstractTensor tokenEmbeddings, int[] attentionMask, int batch, int sequenceLength,
            int hidden, boolean sqrtLength) {
        float[] output = new float[hidden];
        int count = 0;
        for (int token = 0; token < sequenceLength; token++) {
            int row = batch * sequenceLength + token;
            if (attentionMask[row] == 0) {
                continue;
            }
            count++;
            for (int col = 0; col < hidden; col++) {
                output[col] += tokenEmbeddings.get(row, col);
            }
        }
        if (count == 0) {
            return output;
        }
        float divisor = sqrtLength ? (float) Math.sqrt(count) : count;
        for (int col = 0; col < hidden; col++) {
            output[col] /= divisor;
        }
        return output;
    }

    private static float[] weightedMean(AbstractTensor tokenEmbeddings, int[] attentionMask, int batch,
            int sequenceLength, int hidden) {
        float[] output = new float[hidden];
        float weightSum = 0.0f;
        for (int token = 0; token < sequenceLength; token++) {
            int row = batch * sequenceLength + token;
            if (attentionMask[row] == 0) {
                continue;
            }
            float weight = token + 1.0f;
            weightSum += weight;
            for (int col = 0; col < hidden; col++) {
                output[col] += weight * tokenEmbeddings.get(row, col);
            }
        }
        if (weightSum == 0.0f) {
            return output;
        }
        for (int col = 0; col < hidden; col++) {
            output[col] /= weightSum;
        }
        return output;
    }

    private static float[] lastToken(AbstractTensor tokenEmbeddings, int[] attentionMask, int batch,
            int sequenceLength, int hidden) {
        for (int token = sequenceLength - 1; token >= 0; token--) {
            int row = batch * sequenceLength + token;
            if (attentionMask[row] != 0) {
                return row(tokenEmbeddings, row, hidden);
            }
        }
        return new float[hidden];
    }

    private static float[] row(AbstractTensor tokenEmbeddings, int row, int hidden) {
        float[] output = new float[hidden];
        for (int col = 0; col < hidden; col++) {
            output[col] = tokenEmbeddings.get(row, col);
        }
        return output;
    }
}
