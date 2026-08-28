package io.teknek.deliverance.model.diffusiongemma;

import com.google.common.base.Preconditions;

/** Small deterministic helpers for DiffusionGemma generation mechanics. */
public final class DiffusionGemmaGenerationSupport {
    private DiffusionGemmaGenerationSupport() {
    }

    /**
     * Computes generated tokens per decoder forward pass for each batch row.
     *
     * <p>This mirrors Hugging Face DiffusionGemma's secondary generation metric. For each sequence, count all non-pad
     * tokens, subtract the number of prompt tokens that were already present before this generation call, then divide by
     * the number of decoder forward passes used for that row.</p>
     */
    public static float[] computeTokensPerForward(int[][] sequences, int[] decoderForwardPasses,
            int initialInputIdsLength, int padTokenId) {
        Preconditions.checkArgument(sequences.length > 0, "sequences must have at least one batch row");
        Preconditions.checkArgument(decoderForwardPasses.length == sequences.length,
                "decoderForwardPasses length must match batch size");
        Preconditions.checkArgument(initialInputIdsLength >= 0, "initialInputIdsLength must be >= 0");
        float[] tokensPerForward = new float[sequences.length];
        for (int batch = 0; batch < sequences.length; batch++) {
            Preconditions.checkArgument(decoderForwardPasses[batch] > 0,
                    "decoderForwardPasses must be > 0 for batch " + batch);
            int validTokens = 0;
            for (int token : sequences[batch]) {
                if (token != padTokenId) {
                    validTokens++;
                }
            }
            int generatedTokens = validTokens - initialInputIdsLength;
            Preconditions.checkArgument(generatedTokens >= 0,
                    "valid token count must be >= initialInputIdsLength for batch " + batch);
            tokensPerForward[batch] = (float) generatedTokens / decoderForwardPasses[batch];
        }
        return tokensPerForward;
    }
}
