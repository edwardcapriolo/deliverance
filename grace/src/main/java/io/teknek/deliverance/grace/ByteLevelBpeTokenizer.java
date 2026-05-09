package io.teknek.deliverance.grace;

import io.teknek.deliverance.grace.models.TokenizerConfig;

import java.math.BigInteger;
import java.util.*;

/**
 * Generic byte-level BPE tokenizer path for GPT/Qwen/Llama-style fast tokenizers.
 *
 * This intentionally keeps the default byte-level encode/decode hooks from
 * {@link PreTrainedTokenizerBase} / {@link PreTrainedTokenizer} and should not be used for
 * Gemma-style "space to ▁" tokenizers.
 */
public class ByteLevelBpeTokenizer extends PreTrainedTokenizer {
    private final Map<String, Integer> vocab;
    private final Map<Integer, String> idToToken;
    private final TokenizerConfig tokenizerConfig;
    private final BytePairEncodingModel bytePairEncodingModel;

    public ByteLevelBpeTokenizer(Map<String, String> modelSpecificSpecialTokens,
                                 Optional<BigInteger> maxLen,
                                 Optional<PaddingSide> paddingSide,
                                 Optional<TruncationSide> truncationSide,
                                 Optional<Boolean> cleanUpTokenizationSpaces,
                                 Optional<Boolean> splitSpecialTokens,
                                 Optional<Object> backend,
                                 Optional<List<Object>> filesLoaded,
                                 Map<String, Integer> vocab,
                                 SortedMap<Integer, AddedToken> addedTokenMap,
                                 TokenizerConfig tokenizerConfig,
                                 BytePairEncodingModel bytePairEncodingModel) {
        super(modelSpecificSpecialTokens, maxLen, paddingSide, truncationSide, cleanUpTokenizationSpaces,
                splitSpecialTokens, backend, filesLoaded, Optional.of(tokenizerConfig), Optional.of(addedTokenMap));
        this.vocab = Collections.unmodifiableMap(new LinkedHashMap<>(vocab));
        this.idToToken = Collections.unmodifiableMap(buildIdToToken(vocab));
        this.tokenizerConfig = tokenizerConfig;
        this.bytePairEncodingModel = bytePairEncodingModel;
    }

    @Override
    protected Map<String, Integer> getBaseVocab() {
        return vocab;
    }

    @Override
    protected Map<Integer, String> getBaseIdToToken() {
        return idToToken;
    }

    @Override
    public Optional<String> chatTemplate() {
        return Optional.ofNullable(tokenizerConfig.chatTemplate());
    }

    @Override
    protected Optional<BytePairEncodingModel> bytePairEncodingModel() {
        return Optional.ofNullable(bytePairEncodingModel);
    }

    private Map<Integer, String> buildIdToToken(Map<String, Integer> sourceVocab) {
        Map<Integer, String> reverse = new LinkedHashMap<>(sourceVocab.size());
        for (Map.Entry<String, Integer> entry : sourceVocab.entrySet()) {
            reverse.put(entry.getValue(), entry.getKey());
        }
        return reverse;
    }
}
