package io.teknek.deliverance.model.embeddinggemma2;

import io.teknek.deliverance.tensor2.BatchDotProduct;
import io.teknek.deliverance.tensor2.Lighter;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.Objects;

/** Computes masked QK attention scores for head-packed TensorRefs. */
public final class EmbeddingGemma2AttentionScoreOperation {
    private final int queryHeads;
    private final int keyValueHeads;
    private final float scale;

    /** Creates a score operation with GQA head mapping and attention scale. */
    public EmbeddingGemma2AttentionScoreOperation(int queryHeads, int keyValueHeads, float scale) {
        if (queryHeads <= 0 || keyValueHeads <= 0 || queryHeads % keyValueHeads != 0
                || !Float.isFinite(scale)) {
            throw new IllegalArgumentException("Invalid EmbeddingGemma 2 attention score configuration");
        }
        this.queryHeads = queryHeads;
        this.keyValueHeads = keyValueHeads;
        this.scale = scale;
    }

    /**
     * Computes one score matrix per query head. Query and key heads are shaped
     * {@code [tokens, headWidth]}; score matrices are {@code [tokens, tokens]}.
     */
    public void apply(TensorRef[] queries, TensorRef[] keys, TensorRef[] scores,
                      String layerType, int slidingWindow, int[] attentionMask, Lighter lighter) {
        Objects.requireNonNull(queries, "queries");
        Objects.requireNonNull(keys, "keys");
        Objects.requireNonNull(scores, "scores");
        Objects.requireNonNull(layerType, "layerType");
        Objects.requireNonNull(lighter, "lighter");
        if (queries.length != queryHeads || scores.length != queryHeads || keys.length != keyValueHeads) {
            throw new IllegalArgumentException("Attention head array counts do not match configuration");
        }
        for (int head = 0; head < queryHeads; head++) {
            applyHead(queries[head], keys[head / (queryHeads / keyValueHeads)], scores[head], layerType,
                    slidingWindow, attentionMask, lighter);
        }
    }

    /** Computes one query-head score matrix, allowing independent head scheduling. */
    public void applyHead(TensorRef query, TensorRef key, TensorRef score, String layerType,
            int slidingWindow, int[] attentionMask, Lighter lighter) {
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(score, "score");
        Objects.requireNonNull(layerType, "layerType");
        Objects.requireNonNull(lighter, "lighter");
            if (query.dims() != 2 || key.dims() != 2 || score.dims() != 2
                    || query.shape().first() != score.shape().first()
                    || key.shape().first() != score.shape().last()
                    || query.shape().last() != key.shape().last()) {
                throw new IllegalArgumentException("Invalid attention score tensor shapes");
            }
            lighter.batchDotProduct(new BatchDotProduct().result(score).a(query).b(key)
                    .aColumnOffset(0).bColumnOffset(0).columnLength(query.shape().last())
                    .resultRowOffset(0).bRowOffset(0).rowChunkSize(key.shape().first()));
            if (scale != 1.0f) {
                lighter.multiplyAll(scale, score);
            }
            for (int queryPosition = 0; queryPosition < score.shape().first(); queryPosition++) {
                for (int keyPosition = 0; keyPosition < score.shape().last(); keyPosition++) {
                    if (!EmbeddingGemma2AttentionMask.allows(layerType, queryPosition, keyPosition,
                            slidingWindow, attentionMask)) {
                        score.set(-Float.MAX_VALUE, queryPosition, keyPosition);
                    }
                }
            }
    }
}
