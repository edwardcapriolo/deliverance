package io.teknek.deliverance.model.embeddinggemma2;

/** Pure attention-visibility rules for EmbeddingGemma 2 text layers. */
public final class EmbeddingGemma2AttentionMask {
    private EmbeddingGemma2AttentionMask() {
    }

    /**
     * Returns whether a query position may attend to a key position.
     *
     * <p>EmbeddingGemma 2 attention is bidirectional. Full-attention layers
     * have no distance restriction. Sliding-attention layers use a symmetric
     * radius, not a causal look-back window. Padding is applied to keys only;
     * outputs for padded query positions are ignored by downstream pooling.</p>
     *
     * @param layerType {@code full_attention} or {@code sliding_attention}
     * @param queryPosition query position
     * @param keyPosition key position
     * @param slidingWindow inclusive sliding radius
     * @param attentionMask optional key-valid mask, where zero means padding
     * @return whether the key is visible to the query
     */
    public static boolean allows(String layerType, int queryPosition, int keyPosition,
                                 int slidingWindow, int[] attentionMask) {
        if (!"full_attention".equals(layerType) && !"sliding_attention".equals(layerType)) {
            throw new IllegalArgumentException("Unsupported EmbeddingGemma 2 attention layer type: " + layerType);
        }
        if (queryPosition < 0 || keyPosition < 0
                || (attentionMask != null && keyPosition >= attentionMask.length)) {
            throw new IllegalArgumentException("Attention position is out of bounds");
        }
        if (attentionMask != null && attentionMask[keyPosition] == 0) {
            return false;
        }
        if ("full_attention".equals(layerType)) {
            return true;
        }
        if (slidingWindow < 0) {
            throw new IllegalArgumentException("Sliding window must be non-negative");
        }
        return Math.abs(queryPosition - keyPosition) <= slidingWindow;
    }
}
