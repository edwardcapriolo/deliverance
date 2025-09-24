package io.teknek.deliverance.model.llama;

/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */



import io.teknek.deliverance.BytePairEncodingTokenizer;

import java.nio.file.Path;
import java.util.Optional;
import java.util.stream.Collectors;

public class LlamaTokenizer extends BytePairEncodingTokenizer {
    static final String SPIECE_UNDERLINE = "▁";

    private final int byteFallbackEncodingOffset;

    public LlamaTokenizer(Path modelRoot) {
        super(modelRoot);
        this.byteFallbackEncodingOffset = this.getModel().vocabLookup.getOrDefault("<0x00>", -1L).intValue();
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
/*

    @Override
    protected String postProcess(String sentence) {
        return sentence.stripLeading();
    }

    @Override
    protected String postProcessToken(String decoded) {
        if (decoded == null) decoded = tokenizerModel.unkToken;

        decoded = decoded.replaceAll("</?s>", "");
        decoded = decoded.replaceAll(SPIECE_UNDERLINE, " ");

        if (tokenizerModel.isLegacy() && !tokenizerModel.byteFallback) {
            decoded = decoded.codePoints()
                    .map(c -> getAlteredBytes().inverse().getOrDefault(c, c))
                    .mapToObj(Character::toString)
                    .collect(Collectors.joining());
        }

        return decoded;
    }*/
}