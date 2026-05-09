package io.teknek.deliverance.grace;

import io.teknek.deliverance.grace.models.TokenizerConfig;
import io.teknek.dysfx.multiple.Tuple2;

import java.math.BigInteger;
import java.util.*;

public abstract class PreTrainedTokenizer extends PreTrainedTokenizerBase {
    public PreTrainedTokenizer(Map<String, String> modelSpecificSpecialTokens,
                               Optional<BigInteger> maxLen,
                               Optional<PaddingSide> paddingSide,
                               Optional<TruncationSide> truncationSide,
                               Optional<Boolean> cleanUpTokenizationSpaces,
                               Optional<Boolean> splitSpecialTokens,
                               Optional<Object> backend,
                               Optional<List<Object>> filesLoaded,
                               Optional<TokenizerConfig> tokenizerConfig,
                               Optional<SortedMap<Integer, AddedToken>> addedTokensDecoder) {
        super(modelSpecificSpecialTokens, maxLen, paddingSide, truncationSide, cleanUpTokenizationSpaces,
                splitSpecialTokens, backend, filesLoaded, tokenizerConfig, addedTokensDecoder);
    }

    public static boolean isWhitespace(int c) {
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            return true;
        }
        return Character.isWhitespace(c);
    }

    public static boolean isControl(byte b) {
        return isControl((int) b);
    }

    public static boolean isControl(int c) {
        if (c == '\t' || c == '\n' || c == '\r') {
            return false;
        }
        return Character.isISOControl(c);
    }

    public static boolean isEndOfWord(String str) {
        int lastCodepointStartIndex = str.offsetByCodePoints(str.length(), -1);
        int last = str.codePointAt(lastCodepointStartIndex);
        return isControl(last) || isPunctuation(last) || isWhitespace(last);
    }

    public static boolean isStartOfWord(String str) {
        return str.codePoints().findFirst()
                .stream()
                .anyMatch(first -> isControl(first) || isPunctuation(first) || isWhitespace(first));
    }

    public static boolean isPunctuation(int cp) {
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64)
                || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
            return true;
        }
        int type = Character.getType(cp);
        return type == Character.DASH_PUNCTUATION || type == Character.START_PUNCTUATION
                || type == Character.END_PUNCTUATION || type == Character.CONNECTOR_PUNCTUATION
                || type == Character.OTHER_PUNCTUATION || type == Character.INITIAL_QUOTE_PUNCTUATION
                || type == Character.FINAL_QUOTE_PUNCTUATION;
    }

    public int getVocabSize() {
        return vocabSize();
    }

    public String decode(TokenIds tokenIds,
                         boolean skipSpecialTokens,
                         boolean cleanUpTokenizationSpaces,
                         boolean spacesBetweenSpecialTokens,
                         boolean useSourceTokenizer) {
        int[] ids = tokenIds.asArray();
        Set<Integer> specialIds = allSpecialIdSet();
        StringBuilder result = new StringBuilder();
        StringBuilder regularTokenBuffer = new StringBuilder();
        boolean previousWasSpecial = false;

        for (int id : ids) {
            if (skipSpecialTokens && specialIds.contains(id)) {
                continue;
            }

            Optional<String> token = idToToken(id);
            if (token.isEmpty()) {
                continue;
            }

            if (specialIds.contains(id)) {
                flushRegularTokens(result, regularTokenBuffer);
                if (spacesBetweenSpecialTokens && previousWasSpecial && !result.isEmpty()) {
                    result.append(' ');
                }
                result.append(token.orElseThrow());
                previousWasSpecial = true;
                continue;
            }

            regularTokenBuffer.append(token.orElseThrow());
            previousWasSpecial = false;
        }

        flushRegularTokens(result, regularTokenBuffer);
        String decoded = result.toString();
        if (cleanUpTokenizationSpaces) {
            decoded = cleanUpTokenization(decoded);
        }
        return decoded;
    }

    public Tokens convertIdsToTokens(TokenIds tokenIds, boolean skipSpecialTokens) {
        return convertIdsToTokens(tokenIds, Optional.of(skipSpecialTokens));
    }

    public static String cleanUpTokenization(String text) {
        return text.replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" n't", "n't")
                .replace(" 'm", "'m")
                .replace(" 's", "'s")
                .replace(" 've", "'ve")
                .replace(" 're", "'re")
                .replace(" 'd", "'d")
                .replace(" 'll", "'ll");
    }

    protected String decodeRegularTokens(String encoded) {
        return ByteLevelCodec.decode(encoded);
    }

    private void flushRegularTokens(StringBuilder result, StringBuilder regularTokenBuffer) {
        if (regularTokenBuffer.isEmpty()) {
            return;
        }
        result.append(decodeRegularTokens(regularTokenBuffer.toString()));
        regularTokenBuffer.setLength(0);
    }

    static class PreTokenizedInput extends ArrayList<String> {
    }

    static class EncodedInput extends ArrayList<Integer> {
    }

    static class TextInputPair extends Tuple2<String, String> {
        public TextInputPair(String s, String s2) {
            super(s, s2);
        }
    }

    static class EncodedInputPair extends Tuple2<List<Integer>, List<Integer>> {
        public EncodedInputPair(List<Integer> integers, List<Integer> integers2) {
            super(integers, integers2);
        }
    }

    static class PreTokenizedInputPair extends Tuple2<List<String>, List<String>> {
        public PreTokenizedInputPair(List<String> strings, List<String> strings2) {
            super(strings, strings2);
        }
    }
}
