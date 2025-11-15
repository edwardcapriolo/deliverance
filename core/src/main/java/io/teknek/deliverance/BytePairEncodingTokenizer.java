package io.teknek.deliverance;

import com.google.common.collect.BiMap;
import com.google.common.collect.ImmutableBiMap;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tokenizer.Tokenizer;
import io.teknek.deliverance.tokenizer.TokenizerModel;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

public abstract class BytePairEncodingTokenizer implements Tokenizer {

    protected final ImmutableBiMap<Integer, Integer> alteredBytes;
    protected final TokenizerModel tokenizerModel;
    protected final PromptSupport promptSupport;

    //this buffer here looks strange
    //protected final ByteBuffer decodeBuffer = ByteBuffer.allocate(4);

    public BytePairEncodingTokenizer(Path modelRoot){
        {
            // https://github.com/openai/gpt-2/blob/master/src/encoder.py#L19
            Map<Integer, Integer> tmpAlteredBytes = new HashMap<>();
            int i = 0;
            for (int c = 0; c < 256; c++) {
                if ((c < '!' || c > '~') && (c < '¡' || c > '¬') && (c < '®' || c > 'ÿ')) {
                    int codepoint = (i++ + 256);
                    tmpAlteredBytes.put(c, codepoint);
                }
            }
            alteredBytes = ImmutableBiMap.copyOf(tmpAlteredBytes);
        }
        File tokenizerFile = modelRoot.resolve("tokenizer.json").toFile();
        File tokenizerConfigFile = modelRoot.resolve("tokenizer_config.json").toFile();
        this.tokenizerModel = TokenizerModel.load(tokenizerFile, tokenizerConfigFile);
        this.promptSupport = new PromptSupport(tokenizerModel);
    }

    @Override
    public List<String> tokenize(String sentence) {
        if (sentence.isEmpty()){
            return Collections.emptyList();
        }

        List<String> sentencePieces = new ArrayList<>();
        if (tokenizerModel.getAddedTokenPattern() != null) {
            // Split the sentence into pieces using the added token pattern
            // Any non-added token is split into pieces using the pre-tokenizer
            String[] pieces = TokenizerModel.split(tokenizerModel.getAddedTokenPattern(), sentence, 0, true);
            for (String piece : pieces) {
                if (!piece.isEmpty()) {
                    if (tokenizerModel.getAddedTokens().containsKey(piece)) {
                        sentencePieces.add(piece);
                    } else if (tokenizerModel.getPreTokenizer() != null) {
                        sentencePieces.addAll(tokenizerModel.getPreTokenizer().pretokenize(piece));
                    } else {
                        sentencePieces.add(piece);
                    }
                }
            }
        } else if (tokenizerModel.getPreTokenizer() != null) {
            sentencePieces.addAll(tokenizerModel.getPreTokenizer().pretokenize(sentence));
        } else {
            sentencePieces.add(sentence);
        }
        return sentencePieces;
    }

    @Override
    public long[] encode(String sentence) {
        List<String> sentencePieces = tokenize(sentence);
        List<Long> tokens = new ArrayList<>(sentencePieces.size());
        int[] codes = sentence.codePoints().toArray();
        for (int i = 0; i < codes.length; i++) {
            String c = Character.toString(codes[i]);
            Long id = tokenizerModel.vocabLookup.get(c);
            if (id != null) {
                tokens.add(id);
            } else {
                if (tokenizerModel.byteFallback) {
                    String code = Character.toString(codes[i]);
                    byte[] chars = code.getBytes(StandardCharsets.UTF_8);
                    for (int k = 0; k < chars.length; k++) {
                        long token = encodeCharacterAsToken(chars[k]);
                        tokens.add(token);
                    }
                } else {
                    if (tokenizerModel.unkToken != null) {
                        tokens.add(tokenizerModel.vocabLookup.get(tokenizerModel.unkToken));
                    }
                }
            }
        }

        //todo this looks very innefficient
        // merge the best consecutive tuple each iteration,
        // until we can't find any more pairs to merge
        while (true) {
            long bestId = -1;
            long bestIdx = -1;
            long bestRank = Long.MAX_VALUE;

            for (int i = 0; i < tokens.size() - 1; i++) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                String token1 = decodeInternal(tokens.get(i));
                String token2 = decodeInternal(tokens.get(i + 1));

                String merge2 = String.format("%s %s", token1, token2);
                String merge3 = String.format("%s%s", token1, token2);

                if (tokenizerModel.merges.containsKey(merge2)) {
                    Long id = tokenizerModel.vocabLookup.get(merge3);
                    if (id != null) {
                        // Check if this merge has a better rank (i.e., lower rank number)
                        long rank = tokenizerModel.merges.get(merge2);
                        if (rank < bestRank) {
                            // this merge pair exists in vocab! record its position
                            bestId = id;
                            bestIdx = i;
                            bestRank = rank;
                        }
                    }
                }
            }

            if (bestIdx == -1) {
                break; // we couldn't find any more pairs to merge, so we're done
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens.set((int) bestIdx, bestId);
            // delete token at position best_idx+1, shift the entire sequence back 1
            tokens.remove((int) bestIdx + 1);
        }


        return tokens.stream().mapToLong(s -> s).toArray();
    }

    protected abstract long encodeCharacterAsToken(byte c);

    protected abstract Optional<Character> maybeDecodeTokenAsCharacter(long id);

    protected String decodeInternal(long id) {
        return maybeDecodeTokenAsCharacter(id).map(Object::toString).orElseGet(() -> {
            //why not getDefault?
            String s = tokenizerModel.vocabLookup.inverse().get(id);
            if (s == null) {
                s = tokenizerModel.unkToken;
            }
            return s;
        });
    }

    protected String postProcess(String sentence) {
        return sentence;
    }

    @Override
    public String decode(long id) {
        return maybeDecodeTokenAsCharacter(id).map(c -> {
            ByteBuffer decodeBuffer = ByteBuffer.allocate(4);
            // We have a continuation byte or are buffering them
            if (Character.isUnicodeIdentifierPart(c) || decodeBuffer.remaining() < 4) {
                decodeBuffer.put((byte) c.charValue());

                // Unicode symbol is ready
                if (decodeBuffer.remaining() == 0) {
                    String s = new String(decodeBuffer.array());
                    decodeBuffer.rewind();
                    return s;
                }

                return "";
            }
            return Character.toString(c);
        }).orElseGet(() -> postProcessToken(tokenizerModel.vocabLookup.inverse().get(id)));
    }



    @Override
    public String decode(long [] ids) {
        return postProcess(Arrays.stream(ids).mapToObj(this::decode).collect(Collectors.joining()));
    }

    @Override
    public TokenizerModel getModel() {
        return tokenizerModel;
    }

    public BiMap<Integer, Integer> getAlteredBytes(){
        return alteredBytes;
    }

    @Override
    public Optional<PromptSupport> promptSupport() {
        return tokenizerModel.getPromptTemplates().isPresent() ? Optional.of(promptSupport) : Optional.empty();
    }

    protected String postProcessToken(String decoded) {
        if (decoded == null) {
            decoded = tokenizerModel.unkToken;
        }
        return decoded;
    }
}
