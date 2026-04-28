package io.teknek.deliverance.grace.models;

import io.teknek.deliverance.grace.PaddingSide;
import io.teknek.deliverance.grace.TokenizerUtils;
import io.teknek.deliverance.grace.TruncationSide;

import java.math.BigInteger;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public record TokenizerConfig(
        String chatTemplate,
        String tokenizerClass,
        boolean cleanUpTokenizationSpaces,
        boolean splitSpecialTokens,
        BigInteger modelMaxLength,
        PaddingSide paddingSide,
        TruncationSide truncationSide,
        Map<String, String> specialTokensMap,
        List<String> additionalSpecialTokens) {

    public TokenizerConfig {
        modelMaxLength = modelMaxLength == null ? TokenizerUtils.VERY_LARGE_INTEGER : modelMaxLength;
        paddingSide = paddingSide == null ? PaddingSide.RIGHT : paddingSide;
        truncationSide = truncationSide == null ? TruncationSide.RIGHT : truncationSide;
        specialTokensMap = Map.copyOf(new LinkedHashMap<>(specialTokensMap == null ? Map.of() : specialTokensMap));
        additionalSpecialTokens = List.copyOf(additionalSpecialTokens == null ? List.of() : additionalSpecialTokens);
    }

    public static TokenizerConfig empty() {
        return new TokenizerConfig(
                null,
                null,
                false,
                false,
                TokenizerUtils.VERY_LARGE_INTEGER,
                PaddingSide.RIGHT,
                TruncationSide.RIGHT,
                Map.of(),
                List.of());
    }
}
