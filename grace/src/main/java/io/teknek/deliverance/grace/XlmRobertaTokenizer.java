package io.teknek.deliverance.grace;

import io.teknek.deliverance.grace.models.TokenizerConfig;

import java.math.BigInteger;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.SortedMap;

/** XLM-R tokenizer backed by the model's Unigram vocabulary and Metaspace convention. */
public class XlmRobertaTokenizer extends PreTrainedTokenizer {
    private final Map<String, Integer> vocab;
    private final Map<Integer, String> idToToken;
    private final TokenizerConfig tokenizerConfig;
    private final UnigramModel unigram;

    public XlmRobertaTokenizer(Map<String, String> modelSpecificSpecialTokens,
            Optional<BigInteger> maxLen, Optional<PaddingSide> paddingSide, Optional<TruncationSide> truncationSide,
            Optional<Boolean> cleanUpTokenizationSpaces, Optional<Boolean> splitSpecialTokens, Optional<Object> backend,
            Optional<List<Object>> filesLoaded, Map<String, Integer> vocab,
            SortedMap<Integer, AddedToken> addedTokenMap, TokenizerConfig tokenizerConfig, UnigramModel unigram) {
        super(modelSpecificSpecialTokens, maxLen, paddingSide, truncationSide, cleanUpTokenizationSpaces,
                splitSpecialTokens, backend, filesLoaded, Optional.of(tokenizerConfig), Optional.of(addedTokenMap));
        this.vocab = Collections.unmodifiableMap(new LinkedHashMap<>(vocab));
        this.idToToken = reverse(vocab);
        this.tokenizerConfig = tokenizerConfig;
        this.unigram = unigram;
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
    protected int[] buildInputsWithSpecialTokens(int[] inputIds) {
        int bos = bosTokenId().orElseThrow();
        int eos = eosTokenId().orElseThrow();
        int[] result = new int[inputIds.length + 2];
        result[0] = bos;
        System.arraycopy(inputIds, 0, result, 1, inputIds.length);
        result[result.length - 1] = eos;
        return result;
    }

    @Override
    public Encoding encode(String text, EncodeOptions options) {
        TokenIds tokenIds = convertTokensToIds(tokenize(text));
        int[] content = tokenIds.asArray();
        if (options.truncation() != null && options.truncation().maxLength() < content.length + 2
                && options.addSpecialTokens()) {
            int contentLength = Math.max(0, options.truncation().maxLength() - 2);
            content = java.util.Arrays.copyOf(content, contentLength);
        }
        int[] inputIds = options.addSpecialTokens() ? buildInputsWithSpecialTokens(content) : content;
        Encoding encoding = createEncoding(inputIds, inputIds.length);
        if (options.truncation() != null && inputIds.length > options.truncation().maxLength()) {
            encoding = truncate(new TokenIds(inputIds), options.truncation());
        }
        if (options.padding() != null) {
            encoding = pad(new TokenIds(encoding.inputIds()), options.padding());
        }
        return encoding;
    }

    @Override
    protected List<String> tokenizeSegment(String text) {
        String normalized = Normalizer.normalize(text, Normalizer.Form.NFKC)
                .replaceAll("\\s+", " ")
                .strip();
        if (normalized.isEmpty()) {
            return List.of();
        }

        List<String> tokens = new ArrayList<>();
        for (String word : normalized.split(" ")) {
            tokens.addAll(segment("▁" + word));
        }
        return tokens;
    }

    private List<String> segment(String text) {
        int[] offsets = text.codePoints().toArray();
        float[] best = new float[offsets.length + 1];
        int[] previous = new int[offsets.length + 1];
        String[] bestToken = new String[offsets.length + 1];
        java.util.Arrays.fill(best, Float.NEGATIVE_INFINITY);
        best[0] = 0.0f;

        for (int start = 0; start < offsets.length; start++) {
            if (best[start] == Float.NEGATIVE_INFINITY) {
                continue;
            }
            StringBuilder candidate = new StringBuilder();
            for (int end = start; end < offsets.length; end++) {
                candidate.appendCodePoint(offsets[end]);
                String token = candidate.toString();
                Float score = unigram.scores().get(token);
                Integer id = unigram.vocab().get(token);
                if (id != null && score != null && best[start] + score > best[end + 1]) {
                    best[end + 1] = best[start] + score;
                    previous[end + 1] = start;
                    bestToken[end + 1] = token;
                }
            }
            if (best[start + 1] == Float.NEGATIVE_INFINITY) {
                best[start + 1] = best[start] + unigram.scores().getOrDefault(
                        idToToken.get(unigram.unknownId()), -100.0f);
                previous[start + 1] = start;
                bestToken[start + 1] = idToToken.get(unigram.unknownId());
            }
        }

        List<String> result = new ArrayList<>();
        for (int cursor = offsets.length; cursor > 0; cursor = previous[cursor]) {
            result.add(bestToken[cursor]);
        }
        Collections.reverse(result);
        return result;
    }

    @Override
    protected String decodeRegularTokens(String encoded) {
        return encoded.replace('▁', ' ');
    }

    @Override
    public String decode(TokenIds tokenIds, boolean skipSpecialTokens, boolean cleanUpTokenizationSpaces,
            boolean spacesBetweenSpecialTokens, boolean useSourceTokenizer) {
        return super.decode(tokenIds, skipSpecialTokens, cleanUpTokenizationSpaces,
                spacesBetweenSpecialTokens, useSourceTokenizer).stripLeading();
    }

    private static Map<Integer, String> reverse(Map<String, Integer> source) {
        Map<Integer, String> reverse = new LinkedHashMap<>();
        source.forEach((token, id) -> reverse.put(id, token));
        return Collections.unmodifiableMap(reverse);
    }
}
