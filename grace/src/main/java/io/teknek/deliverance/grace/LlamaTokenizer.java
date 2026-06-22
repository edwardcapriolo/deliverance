package io.teknek.deliverance.grace;

import io.teknek.deliverance.grace.models.TokenizerConfig;

import java.io.ByteArrayOutputStream;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.SortedMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class LlamaTokenizer extends GemmaTokenizer {
    private static final Pattern BYTE_FALLBACK = Pattern.compile("<0x([0-9A-Fa-f]{2})>");
    private final Set<String> addedTokenContents;

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
        this.addedTokenContents = new HashSet<>();
        for (AddedToken token : addedTokenMap.values()) {
            this.addedTokenContents.add(token.content());
        }
    }

    @Override
    public Tokens tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        boolean firstPlainSegment = true;
        for (String segment : splitByAddedTokens(text)) {
            if (addedTokenContents.contains(segment)) {
                tokens.add(segment);
                firstPlainSegment = false;
                continue;
            }
            tokens.addAll(tokenizeSegment(segment, firstPlainSegment));
            firstPlainSegment = false;
        }
        return new Tokens(tokens.toArray(new String[0]));
    }

    @Override
    protected String normalizeForBpeInput(String text, BytePairEncodingModel model) {
        return normalizeForBpeInput(text, true);
    }

    private String normalizeForBpeInput(String text, boolean prependPrefix) {
        String normalized = text.replace(" ", "▁");
        return prependPrefix && !normalized.startsWith("▁") ? "▁" + normalized : normalized;
    }

    private List<String> tokenizeSegment(String text, boolean prependPrefix) {
        Optional<BytePairEncodingModel> optionalModel = bytePairEncodingModel();
        if (optionalModel.isEmpty()) {
            throw new UnsupportedOperationException("Tokenizer does not have an encode model");
        }
        BytePairEncodingModel model = optionalModel.orElseThrow();
        String normalized = normalizeForBpeInput(text, model);
        if (!prependPrefix && normalized.startsWith("▁") && !text.startsWith(" ")) {
            normalized = normalized.substring(1);
        }
        List<String> tokens = new ArrayList<>();
        for (String piece : pretokenize(normalized, model.preTokenizer())) {
            if (!piece.isEmpty()) {
                tokens.addAll(applyBpe(piece, model));
            }
        }
        return tokens;
    }

    @Override
    protected List<String> applyBpe(String token, BytePairEncodingModel model) {
        String encoded = encodeForBpe(token, model);
        if (model.vocab().containsKey(encoded)) {
            return List.of(encoded);
        }

        List<String> symbols = encoded.codePoints()
                .mapToObj(codePoint -> new String(Character.toChars(codePoint)))
                .collect(java.util.stream.Collectors.toCollection(ArrayList::new));

        while (symbols.size() > 1) {
            int bestRank = Integer.MAX_VALUE;
            int bestIndex = -1;
            for (int index = 0; index < symbols.size() - 1; index++) {
                String pair = symbols.get(index) + " " + symbols.get(index + 1);
                Integer rank = model.mergeRanks().get(pair);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIndex = index;
                }
            }

            if (bestIndex < 0) {
                break;
            }

            List<String> merged = new ArrayList<>(symbols.size() - 1);
            for (int index = 0; index < symbols.size(); index++) {
                if (index == bestIndex) {
                    merged.add(symbols.get(index) + symbols.get(index + 1));
                    index++;
                } else {
                    merged.add(symbols.get(index));
                }
            }
            symbols = merged;
        }

        List<String> output = new ArrayList<>(symbols.size());
        for (String symbol : symbols) {
            if (model.vocab().containsKey(symbol)) {
                output.add(symbol);
            } else {
                addByteFallbackTokens(output, symbol, model);
            }
        }
        return output;
    }

    private void addByteFallbackTokens(List<String> output, String symbol, BytePairEncodingModel model) {
        byte[] bytes = symbol.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
            String fallback = String.format("<0x%02X>", b & 0xff);
            output.add(model.vocab().containsKey(fallback) ? fallback : model.unkToken());
        }
    }

    @Override
    protected String decodeRegularTokens(String encoded) {
        String decoded = decodeByteFallback(super.decodeRegularTokens(encoded));
        if (" ".equals(decoded)) {
            return decoded;
        }
        return decoded.startsWith(" ") ? decoded.substring(1) : decoded;
    }

    public static String decodeByteFallback(String value) {
        StringBuilder result = new StringBuilder();
        int cursor = 0;
        Matcher matcher = BYTE_FALLBACK.matcher(value);
        while (matcher.find(cursor)) {
            result.append(value, cursor, matcher.start());
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            int nextCursor;
            while (true) {
                bytes.write(Integer.parseInt(matcher.group(1), 16));
                nextCursor = matcher.end();
                if (!matcher.find(nextCursor) || matcher.start() != nextCursor) {
                    break;
                }
            }
            result.append(bytes.toString(StandardCharsets.UTF_8));
            cursor = nextCursor;
        }
        result.append(value, cursor, value.length());
        return result.toString();
    }
}
