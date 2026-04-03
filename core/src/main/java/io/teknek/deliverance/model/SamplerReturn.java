package io.teknek.deliverance.model;

import java.util.Objects;
import java.util.Optional;
import java.util.PriorityQueue;

public class SamplerReturn {
    final int token;
    final Optional<PriorityQueue<IndexValueToken>> topNLogProbs;

    public SamplerReturn(int token) {
        this.token = token;
        topNLogProbs = Optional.empty();
    }
    public SamplerReturn(int token, PriorityQueue<IndexValueToken> topNLogProbs) {
        this.token = token;
        this.topNLogProbs = Optional.of(topNLogProbs);
    }

    public int getToken() {
        return token;
    }

    public Optional<PriorityQueue<IndexValueToken>> getTopNLogProbs() {
        return topNLogProbs;
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || getClass() != o.getClass()) return false;
        SamplerReturn that = (SamplerReturn) o;
        return getToken() == that.getToken() && Objects.equals(getTopNLogProbs(), that.getTopNLogProbs());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getToken(), getTopNLogProbs());
    }

    @Override
    public String toString() {
        return "SamplerReturn{" +
                "token=" + token +
                ", topNLogProbs=" + topNLogProbs +
                '}';
    }
}
