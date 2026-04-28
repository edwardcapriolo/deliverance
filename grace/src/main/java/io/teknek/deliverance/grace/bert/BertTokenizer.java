package io.teknek.deliverance.grace.bert;

import io.teknek.deliverance.grace.AddedToken;
import io.teknek.deliverance.grace.BytePairEncodingModel;
import io.teknek.deliverance.grace.PaddingSide;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.grace.TruncationSide;
import io.teknek.deliverance.grace.models.TokenizerConfig;

import java.math.BigInteger;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.SortedMap;

public class BertTokenizer extends PreTrainedTokenizer {
    private final Map<String, Integer> vocab;
    private final Map<Integer, String> idToToken;
    private final TokenizerConfig tokenizerConfig;
    private final BytePairEncodingModel bytePairEncodingModel;

    public BertTokenizer(Map<String, String> modelSpecificSpecialTokens,
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
    public Optional<String> chatTemplate() {
        return Optional.ofNullable(tokenizerConfig.chatTemplate());
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
