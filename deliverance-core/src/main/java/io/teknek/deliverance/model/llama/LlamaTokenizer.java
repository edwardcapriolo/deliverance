package io.teknek.deliverance.model.llama;
import io.teknek.deliverance.BytePairEncodingTokenizer;

import java.nio.file.Path;
import java.util.Optional;
import java.util.stream.Collectors;

public class LlamaTokenizer extends BytePairEncodingTokenizer {
    static final String SPIECE_UNDERLINE = "▁";

    private final int byteFallbackEncodingOffset;

    public LlamaTokenizer(Path modelRoot) {
        super(modelRoot);
        this.byteFallbackEncodingOffset = getModel().vocabLookup.getOrDefault("<0x00>", -1L).intValue();
    }

    @Override
    protected long encodeCharacterAsToken(byte c) {
        return Byte.toUnsignedLong(c) + Math.max(byteFallbackEncodingOffset, 0);
    }

    @Override
    protected Optional<Character> maybeDecodeTokenAsCharacter(long id) {
        // Handle ascii codes (shifted by N in vocab)
        if (tokenizerModel.byteFallback
                && byteFallbackEncodingOffset > 0
                && id >= byteFallbackEncodingOffset
                && id < 256 + byteFallbackEncodingOffset) {
            char c = (char) (id - byteFallbackEncodingOffset);
            return Optional.of(c);
        }
        return Optional.empty();
    }


    @Override
    public String preProcess(String sentence) {
        if (tokenizerModel.getNormalizer() != null) {
            sentence = tokenizerModel.getNormalizer().normalize(sentence);
        }
        if (tokenizerModel.isLegacy() && !tokenizerModel.byteFallback) {
            sentence = sentence.codePoints()
                    .map(c -> getAlteredBytes().getOrDefault(c, c))
                    .mapToObj(Character::toString)
                    .collect(Collectors.joining());
        }
        return sentence;
    }

}