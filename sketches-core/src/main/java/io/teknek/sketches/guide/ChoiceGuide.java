package io.teknek.sketches.guide;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class ChoiceGuide implements Guide {
    private final Map<String, List<Integer>> encodedChoices;
    private final List<Integer> eosTokens;
    private final List<Integer> acceptedTokens = new ArrayList<>();

    public ChoiceGuide(Map<String, List<Integer>> encodedChoices, List<Integer> eosTokens) {
        this.encodedChoices = Map.copyOf(encodedChoices);
        this.eosTokens = List.copyOf(eosTokens);
    }

    @Override
    public List<Integer> getTokens() {
        Set<Integer> tokens = new LinkedHashSet<>();
        for (List<Integer> choice : encodedChoices.values()) {
            if (!startsWith(choice, acceptedTokens)) {
                continue;
            }
            if (choice.size() == acceptedTokens.size()) {
                tokens.addAll(eosTokens);
            } else {
                tokens.add(choice.get(acceptedTokens.size()));
            }
        }
        return List.copyOf(tokens);
    }

    @Override
    public List<Integer> advance(int tokenId) {
        acceptedTokens.add(tokenId);
        return getTokens();
    }

    @Override
    public boolean isFinished() {
        for (List<Integer> choice : encodedChoices.values()) {
            if (choice.size() == acceptedTokens.size() && startsWith(choice, acceptedTokens)) {
                return true;
            }
        }
        return false;
    }

    private boolean startsWith(List<Integer> choice, List<Integer> prefix) {
        if (choice.size() < prefix.size()) {
            return false;
        }
        for (int i = 0; i < prefix.size(); i++) {
            if (!choice.get(i).equals(prefix.get(i))) {
                return false;
            }
        }
        return true;
    }
}
