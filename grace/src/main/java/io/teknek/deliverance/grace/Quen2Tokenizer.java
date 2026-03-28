package io.teknek.deliverance.grace;


import io.teknek.deliverance.grace.models.TokenizerConfig;

import java.math.BigInteger;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.SortedMap;

public class Quen2Tokenizer extends PreTrainedTokenizer {
    private Map<String,Integer> vocab;
    private SortedMap<Integer, AddedToken> addedTokenMap;
    private TokenizerConfig tokenizerConfig;
    public Quen2Tokenizer(Map<String, String> modelSpecificSpecialTokens,
                          Optional<BigInteger> maxLen,
                          Optional<PaddingSide> paddingSide,
                          Optional<TruncationSide> truncationSide,
                          Optional<Boolean> cleanUpTokenizationSpaces,
                          Optional<Boolean> splitSpecialTokens,
                          Optional<Object> backend,
                          Optional<List<Object>> filesLoaded,

                          Map<String,Integer> vocab, SortedMap<Integer, AddedToken> addedTokenMap, TokenizerConfig tokenizerConfig) {
        super(modelSpecificSpecialTokens, maxLen, paddingSide, truncationSide, cleanUpTokenizationSpaces, splitSpecialTokens, backend, filesLoaded);
        this.vocab = vocab;
        this.addedTokenMap = addedTokenMap;
        this.tokenizerConfig = tokenizerConfig;
    }

    @Override
    public int getVocabSize() {
        return vocab.size();
    }

    @Override
    public int vocabSize() {
        return 0;
    }

    @Override
    public Map<String, Integer> getVocab() {
        return vocab;
    }

    public List<Integer> allSpecialIds(){
        return addedTokenMap.entrySet().stream().filter( (pred) -> pred.getValue().special).map((entry)-> entry.getKey()).toList();
    }

    @Override
    public List<String> allSpecialTokens(){
        return addedTokenMap.entrySet().stream().filter( (pred) -> pred.getValue().special).map((entry)-> entry.getValue().content).toList();
    }
    @Override
    public Optional<String> chatTemplate() {
        return Optional.ofNullable(tokenizerConfig.chatTemplate);
    }
}
