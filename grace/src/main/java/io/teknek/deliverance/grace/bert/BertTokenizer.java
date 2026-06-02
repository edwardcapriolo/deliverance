package io.teknek.deliverance.grace.bert;

import io.teknek.deliverance.grace.AddedToken;
import io.teknek.deliverance.grace.BytePairEncodingModel;
import io.teknek.deliverance.grace.PaddingSide;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.grace.TruncationSide;
import io.teknek.deliverance.grace.models.TokenizerConfig;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.SortedMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;

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

    @Override
    protected List<String> tokenizeSegment(String text) {
        String processed = cleanText(text.toLowerCase().strip());
        if (processed.isEmpty()) {
            return List.of();
        }

        List<String> tokens = new ArrayList<>();
        for (String word : processed.split("\\s+")) {
            splitByPunctuation(word)
                    .map(piece -> piece.length() > 200 ? unkToken().orElse("[UNK]") : piece)
                    .forEach(piece -> tokens.addAll(wordPiece(piece)));
        }
        return tokens;
    }

    @Override
    protected int[] buildInputsWithSpecialTokens(int[] inputIds) {
        OptionalInt cls = tokenToId("[CLS]");
        OptionalInt sep = tokenToId("[SEP]");
        if (cls.isEmpty() || sep.isEmpty()) {
            return inputIds;
        }
        int[] result = new int[inputIds.length + 2];
        result[0] = cls.getAsInt();
        System.arraycopy(inputIds, 0, result, 1, inputIds.length);
        result[result.length - 1] = sep.getAsInt();
        return result;
    }

    @Override
    protected String decodeRegularTokens(String encoded) {
        return Arrays.stream(encoded.split(" "))
                .filter(token -> !token.isEmpty())
                .map(token -> token.startsWith("##") ? token.substring(2) : " " + token)
                .collect(Collectors.joining())
                .strip();
    }

    private List<String> wordPiece(String value) {
        boolean isBad = false;
        List<String> subTokens = new ArrayList<>();
        int start = 0;
        while (start < value.length()) {
            int end = value.length();
            String current = null;
            while (start < end) {
                String candidate = value.substring(start, end);
                if (start > 0) {
                    candidate = "##" + candidate;
                }
                if (tokenToId(candidate).isPresent()) {
                    current = candidate;
                    break;
                }
                end--;
            }
            if (current == null) {
                isBad = true;
                break;
            }
            subTokens.add(current);
            start = end;
        }
        return isBad ? List.of(unkToken().orElse("[UNK]")) : subTokens;
    }

    private String cleanText(String value) {
        return value.codePoints()
                .map(c -> c == 0 || c == 0xfffd || PreTrainedTokenizer.isControl(c) ? -1 : Character.isWhitespace(c) ? ' ' : c)
                .filter(c -> c != -1)
                .mapToObj(Character::toString)
                .collect(Collectors.joining());
    }

    private Stream<String> splitByPunctuation(String value) {
        List<String> result = new ArrayList<>();
        int start = 0;
        for (int offset = 0; offset < value.length();) {
            int codepoint = value.codePointAt(offset);
            if (PreTrainedTokenizer.isPunctuation(codepoint)) {
                if (offset != start) {
                    result.add(value.substring(start, offset));
                }
                result.add(value.substring(offset, offset + Character.charCount(codepoint)));
                start = offset + Character.charCount(codepoint);
            }
            offset += Character.charCount(codepoint);
        }
        if (start != value.length()) {
            result.add(value.substring(start));
        }
        return result.stream();
    }

    private Map<Integer, String> buildIdToToken(Map<String, Integer> sourceVocab) {
        Map<Integer, String> reverse = new LinkedHashMap<>(sourceVocab.size());
        for (Map.Entry<String, Integer> entry : sourceVocab.entrySet()) {
            reverse.put(entry.getValue(), entry.getKey());
        }
        return reverse;
    }
}
