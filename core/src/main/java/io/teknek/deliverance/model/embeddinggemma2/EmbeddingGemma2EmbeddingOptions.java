package io.teknek.deliverance.model.embeddinggemma2;

/** Optional text-embedding post-processing and prompt controls. */
public record EmbeddingGemma2EmbeddingOptions(String taskPrompt, String promptOverride,
                                               Integer truncateDimension, boolean normalizeEmbeddings) {
    /** Creates the native 768-dimensional normalized output configuration. */
    public static EmbeddingGemma2EmbeddingOptions defaults() {
        return new EmbeddingGemma2EmbeddingOptions(null, null, null, true);
    }

    /** Validates supported native and Matryoshka dimensions. */
    public EmbeddingGemma2EmbeddingOptions {
        if (taskPrompt != null && taskPrompt.isBlank()) {
            throw new IllegalArgumentException("taskPrompt must not be blank");
        }
        if (promptOverride != null && promptOverride.isBlank()) {
            throw new IllegalArgumentException("promptOverride must not be blank");
        }
        if (truncateDimension != null && truncateDimension != 128 && truncateDimension != 256
                && truncateDimension != 512 && truncateDimension != 768) {
            throw new IllegalArgumentException("truncateDimension must be one of 128, 256, 512, or 768");
        }
    }
}
