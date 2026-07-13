package io.teknek.sketches.guide;

import java.util.List;
import java.util.Objects;

/**
 * Stateful guide backed by an {@link Index} transition table.
 *
 * <p>This is the direct regex/FSA guide path: {@link Index#getAllowedTokens(int)} supplies valid next token ids, and
 * {@link Index#getNextState(int, int)} advances the current state after a token is accepted.</p>
 */
public final class IndexGuide implements Guide {
    private final Index index;
    private int state;

    public IndexGuide(Index index) {
        this.index = Objects.requireNonNull(index, "index");
        this.state = index.getInitialState();
    }

    public int getState() {
        return state;
    }

    @Override
    public List<Integer> getTokens() {
        return index.getAllowedTokens(state);
    }

    @Override
    public List<Integer> advance(int tokenId) {
        state = index.getNextState(state, tokenId);
        return getTokens();
    }

    @Override
    public boolean isFinished() {
        return index.isFinalState(state);
    }

    public boolean acceptsTokens(List<Integer> tokenIds) {
        int candidateState = index.getInitialState();
        for (Integer tokenId : tokenIds) {
            if (index.isEosToken(tokenId)) {
                return false;
            }
            try {
                candidateState = index.getNextState(candidateState, tokenId);
            } catch (IllegalArgumentException e) {
                return false;
            }
        }
        return index.isFinalState(candidateState);
    }
}
