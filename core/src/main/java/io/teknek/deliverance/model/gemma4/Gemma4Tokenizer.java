package io.teknek.deliverance.model.gemma4;

import io.teknek.deliverance.BytePairEncodingTokenizer;

import java.nio.file.Path;
import java.util.Optional;

public class Gemma4Tokenizer extends BytePairEncodingTokenizer {
    static final String SPIECE_UNDERLINE = "▁";

    private final int byteFallbackEncodingOffset;

    public Gemma4Tokenizer(Path modelRoot) {
        super(modelRoot);
        this.byteFallbackEncodingOffset = 217;
    }

    @Override
    protected long encodeCharacterAsToken(byte c) {
        return Byte.toUnsignedLong(c) + byteFallbackEncodingOffset;
    }

    @Override
    protected Optional<Character> maybeDecodeTokenAsCharacter(long id) {
        if (getModel().byteFallback && id >= byteFallbackEncodingOffset && id < 256 + byteFallbackEncodingOffset) {
            char c = (char) (id - byteFallbackEncodingOffset);
            return Optional.of(c);
        }
        return Optional.empty();
    }

    @Override
    public String preProcess(String sentence) {
        return sentence.replace(" ", SPIECE_UNDERLINE);
    }

    @Override
    protected String postProcess(String sentence) {
        return sentence.stripLeading();
    }

    @Override
    protected String postProcessToken(String decoded) {
        if (decoded == null) {
            decoded = getModel().unkToken;
        }

        decoded = decoded.replaceAll("</?s>", "");
        decoded = decoded.replaceAll(SPIECE_UNDERLINE, " ");
        return decoded;
    }
}
