package io.teknek.deliverance.model;

import java.util.Objects;

// as we calculate top_logprobs this holds them we havent commit to a model that matches the OIA return shape so
// we hae intermediate forms like this
public class IndexValueToken implements Comparable<IndexValueToken> {
    public int index;
    public float value;
    public String token;
    public float logProb;

    private IndexValueToken() {}
    public IndexValueToken(int index, float value, String token) {
        this.index = index;
        this.value = value;
        this.token = token;
    }

    public float getProbability(){
        return (float) Math.exp(logProb);
    }

    /** To support top log_probs sort on value*/
    @Override
    public int compareTo(IndexValueToken o) {
        if (this.value == o.value) {
            return Integer.compare(this.index, o.index);
        }
        return Float.compare(this.value, o.value);
    }

    @Override
    public String toString() {
        return "IndexValueToken{" +
                "index=" + index +
                ", value=" + value +
                ", token='" + token + '\'' +
                ", logProb=" + logProb +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || getClass() != o.getClass()) return false;
        IndexValueToken that = (IndexValueToken) o;
        return index == that.index && Float.compare(value, that.value) == 0
                && Float.compare(logProb, that.logProb) == 0 && Objects.equals(token, that.token);
    }

    @Override
    public int hashCode() {
        return Objects.hash(index, value, token, logProb);
    }
}
