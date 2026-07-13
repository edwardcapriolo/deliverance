package io.teknek.sketches.guide;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Token vocabulary used by structured generation guides.
 *
 * <p>This is not a tokenizer. It is the text/token-id bridge used after a model tokenizer has already exposed its token
 * ids and decoded token strings. Multiple token ids may decode to the same string, so the forward mapping is
 * {@code String -> List<Integer>}.</p>
 */
public final class Vocabulary {
    private final List<Integer> eosTokenIds;
    private final Map<String, List<Integer>> tokenIdsByText = new LinkedHashMap<>();
    private final Map<Integer, String> textByTokenId = new LinkedHashMap<>();

    public Vocabulary(int eosTokenId, Map<String, List<Integer>> tokens) {
        this(List.of(eosTokenId), tokens);
    }

    public Vocabulary(List<Integer> eosTokenIds, Map<String, List<Integer>> tokens) {
        Objects.requireNonNull(eosTokenIds, "eosTokenIds");
        if (eosTokenIds.isEmpty()) {
            throw new IllegalArgumentException("Vocabulary must have at least one EOS token id");
        }
        this.eosTokenIds = List.copyOf(eosTokenIds);
        Objects.requireNonNull(tokens, "tokens");
        for (Map.Entry<String, List<Integer>> entry : tokens.entrySet()) {
            String token = requireToken(entry.getKey());
            List<Integer> tokenIds = Objects.requireNonNull(entry.getValue(), "token id list for " + token);
            for (Integer tokenId : tokenIds) {
                insert(token, Objects.requireNonNull(tokenId, "token id for " + token));
            }
        }
    }

    public Vocabulary(Vocabulary other) {
        this(other.eosTokenIds, other.tokenIdsByText);
    }

    /**
     * Returns the first EOS token id.
     *
     * <p>This method mirrors Outlines Core's single-EOS API. Use {@link #getEosTokenIds()} when adapting Deliverance
     * models that expose multiple EOS tokens.</p>
     */
    public int getEosTokenId() {
        return eosTokenIds.getFirst();
    }

    public List<Integer> getEosTokenIds() {
        return eosTokenIds;
    }

    public List<Integer> get(String token) {
        requireToken(token);
        List<Integer> tokenIds = tokenIdsByText.get(token);
        return tokenIds == null ? null : List.copyOf(tokenIds);
    }

    public void insert(String token, int tokenId) {
        requireToken(token);
        if (eosTokenIds.contains(tokenId)) {
            throw new IllegalArgumentException("EOS token should not be inserted into Vocabulary");
        }
        tokenIdsByText.computeIfAbsent(token, ignored -> new ArrayList<>()).add(tokenId);
        textByTokenId.put(tokenId, token);
    }

    public void remove(String token) {
        requireToken(token);
        List<Integer> removed = tokenIdsByText.remove(token);
        if (removed == null) {
            return;
        }
        for (Integer tokenId : removed) {
            textByTokenId.remove(tokenId);
        }
    }

    public String tokenText(int tokenId) {
        return textByTokenId.get(tokenId);
    }

    public int size() {
        return textByTokenId.size() + eosTokenIds.size();
    }

    public Map<String, List<Integer>> tokens() {
        Map<String, List<Integer>> copy = new LinkedHashMap<>();
        for (Map.Entry<String, List<Integer>> entry : tokenIdsByText.entrySet()) {
            copy.put(entry.getKey(), List.copyOf(entry.getValue()));
        }
        return Map.copyOf(copy);
    }

    private String requireToken(String token) {
        if (token == null) {
            throw new IllegalArgumentException("Expected a token of type String, got null");
        }
        return token;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Vocabulary that)) {
            return false;
        }
        return eosTokenIds.equals(that.eosTokenIds) && tokenIdsByText.equals(that.tokenIdsByText);
    }

    @Override
    public int hashCode() {
        return Objects.hash(eosTokenIds, tokenIdsByText);
    }
}
