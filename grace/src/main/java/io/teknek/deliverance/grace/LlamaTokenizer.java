package io.teknek.deliverance.grace;

import io.teknek.deliverance.grace.models.TokenizerConfig;

import java.io.ByteArrayOutputStream;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.SortedMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class LlamaTokenizer extends GemmaTokenizer {
    private static final Pattern BYTE_FALLBACK = Pattern.compile("<0x([0-9A-Fa-f]{2})>");

    public LlamaTokenizer(Map<String, String> modelSpecificSpecialTokens,
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
                splitSpecialTokens, backend, filesLoaded, vocab, addedTokenMap, tokenizerConfig, bytePairEncodingModel);
    }

    @Override
    protected String decodeRegularTokens(String encoded) {
        return decodeByteFallback(super.decodeRegularTokens(encoded));
    }

    public static String decodeByteFallback(String value) {
        Matcher matcher = BYTE_FALLBACK.matcher(value);
        StringBuilder result = new StringBuilder();
        int cursor = 0;
        while (matcher.find()) {
            result.append(value, cursor, matcher.start());
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            do {
                bytes.write(Integer.parseInt(matcher.group(1), 16));
                cursor = matcher.end();
            } while (matcher.find() && matcher.start() == cursor);
            result.append(bytes.toString(StandardCharsets.UTF_8));
        }
        result.append(value, cursor, value.length());
        return result.toString();
    }
}
