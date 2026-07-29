package io.teknek.sketches.guide;

import java.util.List;
import java.util.Objects;

/** Request-scoped {@link Guide} backed by a lazy token-transition index. */
public final class LazyIndexGuide implements Guide {
    private final LazyIndex index;
    private int state;

    public LazyIndexGuide(LazyIndex index) {
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
}
