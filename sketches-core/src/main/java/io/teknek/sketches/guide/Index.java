package io.teknek.sketches.guide;

import dk.brics.automaton.Automaton;
import dk.brics.automaton.RegExp;
import dk.brics.automaton.State;
import io.teknek.sketches.SketchesSettings;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Queue;
import java.util.Set;

/**
 * Token transition index for a regular expression and a model vocabulary.
 *
 * <p>The regex automaton operates on text, while generation operates on token ids. This class precomputes transitions of
 * the form {@code state -> token id -> next state} by walking each decoded token string through the regex automaton from
 * each reachable state.</p>
 *
 * <p>Accepting states include transitions for every configured EOS token id. Those EOS transitions loop back to the same
 * accepting state, mirroring the constrained-decoding convention that EOS is allowed once the generated text satisfies
 * the regex.</p>
 */
public final class Index {
    private int initialState;
    private Map<Integer, Map<Integer, Integer>> transitions;
    private Set<Integer> finalStates;
    private Set<Integer> eosTokenIds;

    public Index(String regex, Vocabulary vocabulary) {
        this(regex, vocabulary, SketchesSettings.DEFAULT);
    }

    public Index(String regex, Vocabulary vocabulary, SketchesSettings settings) {
        Objects.requireNonNull(regex, "regex");
        if (regex.length() > settings.maxRegexLength()) {
            throw new IllegalArgumentException("guided_regex length " + regex.length()
                    + " exceeds maxRegexLength " + settings.maxRegexLength());
        }
        initialize(new RegExp(regex).toAutomaton(), vocabulary, settings, "guided_regex");
    }

    public Index(Automaton automaton, Vocabulary vocabulary) {
        this(automaton, vocabulary, SketchesSettings.DEFAULT);
    }

    public Index(Automaton automaton, Vocabulary vocabulary, SketchesSettings settings) {
        initialize(automaton, vocabulary, settings, "guided_grammar");
    }

    private void initialize(Automaton automaton, Vocabulary vocabulary, SketchesSettings settings, String label) {
        Objects.requireNonNull(automaton, "automaton");
        Objects.requireNonNull(vocabulary, "vocabulary");
        Objects.requireNonNull(settings, "settings");
        State initial = automaton.getInitialState();
        this.eosTokenIds = Set.copyOf(vocabulary.getEosTokenIds());

        Map<State, Integer> stateIds = new IdentityHashMap<>();
        Queue<State> queue = new ArrayDeque<>();
        stateIds.put(initial, 0);
        queue.add(initial);

        Map<Integer, Map<Integer, Integer>> builtTransitions = new LinkedHashMap<>();
        Set<Integer> builtFinalStates = new LinkedHashSet<>();
        List<Integer> tokenIds = sortedTokenIds(vocabulary);
        int transitionCount = 0;

        while (!queue.isEmpty()) {
            if (stateIds.size() > settings.maxIndexStates()) {
                throw new IllegalArgumentException(label + " index exceeded maxIndexStates "
                        + settings.maxIndexStates());
            }
            State state = queue.remove();
            int stateId = stateIds.get(state);
            if (state.isAccept()) {
                builtFinalStates.add(stateId);
            }

            Map<Integer, Integer> stateTransitions = new LinkedHashMap<>();
            if (state.isAccept()) {
                for (Integer eosTokenId : vocabulary.getEosTokenIds()) {
                    stateTransitions.put(eosTokenId, stateId);
                    transitionCount = checkedTransitionCount(transitionCount, settings);
                }
            }

            for (Integer tokenId : tokenIds) {
                String tokenText = vocabulary.tokenText(tokenId);
                if (tokenText == null || tokenText.isEmpty()) {
                    continue;
                }
                State next = walk(state, tokenText);
                if (next == null) {
                    continue;
                }
                Integer nextStateId = stateIds.get(next);
                if (nextStateId == null) {
                    nextStateId = stateIds.size();
                    stateIds.put(next, nextStateId);
                    queue.add(next);
                }
                stateTransitions.put(tokenId, nextStateId);
                transitionCount = checkedTransitionCount(transitionCount, settings);
            }
            builtTransitions.put(stateId, Map.copyOf(stateTransitions));
        }

        this.initialState = 0;
        this.transitions = deepCopy(builtTransitions);
        this.finalStates = Set.copyOf(builtFinalStates);
    }

    public int getInitialState() {
        return initialState;
    }

    /**
     * Returns token ids that can be emitted from {@code state} without making the regex impossible to complete.
     *
     * <p>This is the primary query used by guide-backed logits processors. A processor should keep these token ids and
     * mask every other token id. Returned token ids are sorted for deterministic behavior.</p>
     *
     * <p>If {@code state} is accepting, the returned list includes EOS token ids from the vocabulary. If a decoded token
     * advances through multiple regex characters, such as token text {@code "ab"} for regex {@code "ab"}, it is included
     * when the entire token text can be walked through the automaton from {@code state}.</p>
     *
     * @param state index state id
     * @return sorted valid next token ids, or an empty list if the state is unknown or has no valid transitions
     */
    public List<Integer> getAllowedTokens(int state) {
        Map<Integer, Integer> stateTransitions = transitions.get(state);
        if (stateTransitions == null) {
            return List.of();
        }
        List<Integer> allowedTokens = new ArrayList<>(stateTransitions.keySet());
        Collections.sort(allowedTokens);
        return List.copyOf(allowedTokens);
    }

    /**
     * Returns the next state after accepting {@code tokenId} from {@code state}.
     *
     * <p>The token must be one of {@link #getAllowedTokens(int)} for the same state. Unknown states or disallowed tokens
     * are programming errors for guide implementations and result in an exception.</p>
     *
     * @param state current index state id
     * @param tokenId selected token id
     * @return next index state id
     */
    public int getNextState(int state, int tokenId) {
        Map<Integer, Integer> stateTransitions = transitions.get(state);
        if (stateTransitions == null || !stateTransitions.containsKey(tokenId)) {
            throw new IllegalArgumentException("No next state found for state " + state + " and token " + tokenId);
        }
        return stateTransitions.get(tokenId);
    }

    /**
     * Returns whether {@code state} satisfies the regex without needing more non-EOS tokens.
     */
    public boolean isFinalState(int state) {
        return finalStates.contains(state);
    }

    /**
     * Returns all accepting state ids discovered while building this index.
     */
    public Set<Integer> getFinalStates() {
        return finalStates;
    }

    public boolean isEosToken(int tokenId) {
        return eosTokenIds.contains(tokenId);
    }

    /**
     * Returns the complete precomputed transition table.
     *
     * <p>The outer key is state id, the inner key is token id, and the inner value is the next state id.</p>
     */
    public Map<Integer, Map<Integer, Integer>> getTransitions() {
        return transitions;
    }

    public int stateCount() {
        return transitions.size();
    }

    public int transitionCount() {
        int count = 0;
        for (Map<Integer, Integer> stateTransitions : transitions.values()) {
            count += stateTransitions.size();
        }
        return count;
    }

    private State walk(State state, String tokenText) {
        State current = state;
        for (int i = 0; i < tokenText.length(); i++) {
            current = current.step(tokenText.charAt(i));
            if (current == null) {
                return null;
            }
        }
        return current;
    }

    private static List<Integer> sortedTokenIds(Vocabulary vocabulary) {
        List<Integer> tokenIds = new ArrayList<>();
        for (List<Integer> ids : vocabulary.tokens().values()) {
            tokenIds.addAll(ids);
        }
        Collections.sort(tokenIds);
        return tokenIds;
    }

    private static Map<Integer, Map<Integer, Integer>> deepCopy(Map<Integer, Map<Integer, Integer>> source) {
        Map<Integer, Map<Integer, Integer>> copy = new LinkedHashMap<>();
        for (Map.Entry<Integer, Map<Integer, Integer>> entry : source.entrySet()) {
            copy.put(entry.getKey(), Map.copyOf(entry.getValue()));
        }
        return Map.copyOf(copy);
    }

    private static int checkedTransitionCount(int transitionCount, SketchesSettings settings) {
        int next = transitionCount + 1;
        if (next > settings.maxIndexTransitions()) {
            throw new IllegalArgumentException("guided_regex index exceeded maxIndexTransitions "
                    + settings.maxIndexTransitions());
        }
        return next;
    }
}
