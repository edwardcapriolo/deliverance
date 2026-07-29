package io.teknek.sketches.guide;

import dk.brics.automaton.RegExp;
import dk.brics.automaton.State;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Lazily computes token transitions for a regex automaton.
 *
 * <p>This is the preferred runtime index for guided regex/JSON generation. It avoids the eager {@code states * vocab}
 * transition-table build used by {@link Index} and computes transitions only for guide states that are actually queried.</p>
 */
public final class LazyIndex {
    private final List<Integer> eosTokenIds;
    private final Map<State, Integer> stateIds = new IdentityHashMap<>();
    private final Map<Integer, State> statesById = new HashMap<>();
    private final Map<Integer, Map<Integer, Integer>> transitionsByState = new HashMap<>();
    private final TokenTrie tokenTrie;
    private final int initialState;
    private int computedStateCount;
    private int computedTransitionCount;

    public LazyIndex(String regex, Vocabulary vocabulary) {
        Objects.requireNonNull(regex, "regex");
        Objects.requireNonNull(vocabulary, "vocabulary");
        State initial = new RegExp(regex).toAutomaton().getInitialState();
        this.eosTokenIds = List.copyOf(vocabulary.getEosTokenIds());
        this.tokenTrie = TokenTrie.from(vocabulary);
        this.initialState = stateId(initial);
    }

    public int getInitialState() {
        return initialState;
    }

    public synchronized List<Integer> getAllowedTokens(int state) {
        Map<Integer, Integer> transitions = transitionsForState(state);
        if (transitions.isEmpty()) {
            return List.of();
        }
        List<Integer> allowed = new ArrayList<>(transitions.keySet());
        Collections.sort(allowed);
        return List.copyOf(allowed);
    }

    public synchronized int getNextState(int state, int tokenId) {
        Map<Integer, Integer> transitions = transitionsForState(state);
        Integer next = transitions.get(tokenId);
        if (next == null) {
            throw new IllegalArgumentException("No next state found for state " + state + " and token " + tokenId);
        }
        return next;
    }

    public synchronized boolean isFinalState(int state) {
        State automatonState = statesById.get(state);
        return automatonState != null && automatonState.isAccept();
    }

    public synchronized int computedStateCount() {
        return computedStateCount;
    }

    public synchronized int computedTransitionCount() {
        return computedTransitionCount;
    }

    private Map<Integer, Integer> transitionsForState(int state) {
        Map<Integer, Integer> cached = transitionsByState.get(state);
        if (cached != null) {
            return cached;
        }
        State automatonState = statesById.get(state);
        if (automatonState == null) {
            return Map.of();
        }
        Map<Integer, Integer> transitions = new HashMap<>();
        if (automatonState.isAccept()) {
            for (Integer eosTokenId : eosTokenIds) {
                transitions.put(eosTokenId, state);
            }
        }
        tokenTrie.collectTransitions(automatonState, transitions, this::stateId);
        computedStateCount++;
        computedTransitionCount += transitions.size();
        Map<Integer, Integer> immutable = Map.copyOf(transitions);
        transitionsByState.put(state, immutable);
        return immutable;
    }

    private int stateId(State state) {
        Integer existing = stateIds.get(state);
        if (existing != null) {
            return existing;
        }
        int id = stateIds.size();
        stateIds.put(state, id);
        statesById.put(id, state);
        return id;
    }

    private interface StateIdResolver {
        int stateId(State state);
    }

    private static final class TokenTrie {
        private final TokenTrieNode root = new TokenTrieNode();

        static TokenTrie from(Vocabulary vocabulary) {
            TokenTrie trie = new TokenTrie();
            List<Integer> tokenIds = new ArrayList<>();
            for (List<Integer> ids : vocabulary.tokens().values()) {
                tokenIds.addAll(ids);
            }
            Collections.sort(tokenIds);
            for (Integer tokenId : tokenIds) {
                String tokenText = vocabulary.tokenText(tokenId);
                if (tokenText == null || tokenText.isEmpty()) {
                    continue;
                }
                trie.insert(tokenText, tokenId);
            }
            return trie;
        }

        private void insert(String tokenText, int tokenId) {
            TokenTrieNode current = root;
            for (int i = 0; i < tokenText.length(); i++) {
                char c = tokenText.charAt(i);
                current = current.children.computeIfAbsent(c, ignored -> new TokenTrieNode());
            }
            current.tokenIds.add(tokenId);
        }

        void collectTransitions(State automatonState, Map<Integer, Integer> transitions, StateIdResolver resolver) {
            for (Map.Entry<Character, TokenTrieNode> entry : root.children.entrySet()) {
                State next = automatonState.step(entry.getKey());
                if (next != null) {
                    collectTransitions(next, entry.getValue(), transitions, resolver);
                }
            }
        }

        private void collectTransitions(State automatonState, TokenTrieNode trieNode,
                Map<Integer, Integer> transitions, StateIdResolver resolver) {
            if (!trieNode.tokenIds.isEmpty()) {
                int nextState = resolver.stateId(automatonState);
                for (Integer tokenId : trieNode.tokenIds) {
                    transitions.put(tokenId, nextState);
                }
            }
            for (Map.Entry<Character, TokenTrieNode> entry : trieNode.children.entrySet()) {
                State next = automatonState.step(entry.getKey());
                if (next != null) {
                    collectTransitions(next, entry.getValue(), transitions, resolver);
                }
            }
        }
    }

    private static final class TokenTrieNode {
        private final Map<Character, TokenTrieNode> children = new HashMap<>();
        private final List<Integer> tokenIds = new ArrayList<>();
    }
}
