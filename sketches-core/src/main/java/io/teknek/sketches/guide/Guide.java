package io.teknek.sketches.guide;

import java.util.List;

/**
 * Stateful token-level guide for structured generation.
 *
 * <p>A guide represents the current state of a constraint language, such as a choice set, regex-derived automaton, or JSON
 * schema-derived automaton. It exposes the token ids that are valid at the current state and advances when the sampler
 * accepts a token.</p>
 *
 * <p>Guides are request-scoped and mutable. Do not share one guide across concurrent generations. Implementations should
 * keep all constraint state internally and should not depend on Deliverance-specific classes such as logits tensors or
 * response contexts; backend adapters are responsible for translating {@link #getTokens()} into backend-specific masks.</p>
 */
public interface Guide {
    /**
     * Returns token ids allowed from the current guide state.
     *
     * <p>The returned list should be deterministic for the current state. Implementations may return a defensive copy or an
     * immutable view, but callers must not mutate it.</p>
     *
     * @return valid next token ids
     */
    List<Integer> getTokens();

    /**
     * Advances this guide after {@code tokenId} has been selected.
     *
     * <p>The return value is the allowed-token set after advancement. Callers may ignore it and call {@link #getTokens()}
     * later; it is returned for adapters that want to update masks without a second lookup.</p>
     *
     * @param tokenId selected token id
     * @return valid next token ids after accepting {@code tokenId}
     */
    List<Integer> advance(int tokenId);

    /**
     * Reports whether this guide has reached a completed output state.
     *
     * <p>Completion does not necessarily mean generation should stop immediately. Some adapters may choose to allow only EOS
     * after completion, while others may use this method to coordinate an external stop condition.</p>
     *
     * @return true when the accepted token sequence satisfies the guide's target language
     */
    boolean isFinished();
}
