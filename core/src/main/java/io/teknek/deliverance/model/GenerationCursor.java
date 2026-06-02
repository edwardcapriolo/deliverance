package io.teknek.deliverance.model;

import java.util.Arrays;

/**
 * Computes the token ranges and positions used when resuming generation from a prefix KV-cache hit.
 *
 * <p>The cursor is deliberately small and purely about positional bookkeeping. It does not prove that running
 * {@code batchForward(allPromptTokens)} is numerically identical to running {@code batchForward(cachedPrefix)}
 * followed by {@code batchForward(uncachedSuffix)}. That stronger property is usually called batch/chunk invariance
 * and must be established at the model/kernel level.</p>
 *
 * <p>Reference background: Thinking Machines, "Defeating Nondeterminism in LLM Inference",
 * https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/</p>
 *
 * <p>The invariant this class does enforce is simpler: after the full prompt is represented in KV memory, the
 * first generated token is decoded at {@code promptTokens.length}, regardless of how many prompt tokens were
 * restored from cache. Prefix length must not reduce the decode budget.</p>
 */
public final class GenerationCursor {
    private final int[] promptTokens;
    private final int prefixLength;
    private final int startPosition;
    private final int[] tokensToProcess;
    private final int decodeStartPosition;

    private GenerationCursor(int[] promptTokens, int prefixLength) {
        if (prefixLength < 0 || prefixLength > promptTokens.length) {
            throw new IllegalArgumentException("prefixLength must be within promptTokens");
        }
        this.promptTokens = promptTokens;
        this.prefixLength = prefixLength;
        this.startPosition = prefixLength;
        this.tokensToProcess = prefixLength > 0
                ? Arrays.copyOfRange(promptTokens, prefixLength, promptTokens.length)
                : promptTokens;
        this.decodeStartPosition = promptTokens.length;
    }

    public static GenerationCursor from(int[] promptTokens, int prefixLength) {
        return new GenerationCursor(promptTokens, prefixLength);
    }

    public int prefixLength() {
        return prefixLength;
    }

    public int startPosition() {
        return startPosition;
    }

    public int[] tokensToProcess() {
        return tokensToProcess;
    }

    public boolean hasTokensToProcess() {
        return tokensToProcess.length > 0;
    }

    public int replayToken() {
        if (prefixLength == 0) {
            throw new IllegalStateException("No cached prompt token is available to replay");
        }
        return promptTokens[prefixLength - 1];
    }

    public int replayPosition() {
        if (prefixLength == 0) {
            throw new IllegalStateException("No cached prompt position is available to replay");
        }
        return startPosition - 1;
    }

    public int decodeStartPosition() {
        return decodeStartPosition;
    }
}
