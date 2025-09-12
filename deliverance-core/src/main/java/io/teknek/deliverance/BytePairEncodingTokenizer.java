package io.teknek.deliverance;

import com.google.common.collect.BiMap;
import com.google.common.collect.ImmutableBiMap;
import io.teknek.deliverance.tokenizer.Tokenizer;
import io.teknek.deliverance.tokenizer.TokenizerModel;

import java.nio.file.Path;
import java.util.*;

public class BytePairEncodingTokenizer implements Tokenizer {

    private BiMap<Integer, Integer> alteredBytes;
    protected final TokenizerModel tokenizerModel;


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
        java.io.File tokenizerFile = modelRoot.resolve("tokenizer.json").toFile();
        this.tokenizerModel = TokenizerModel.load(tokenizerFile);
    }

    @Override
    public List<String> tokenize(String sentence) {
        if (sentence.isEmpty()){
            return Collections.emptyList();
        }

        /*
        if (model.preTokenizer() == null && model.addedTokenPattern() == null) Collections.singletonList(sentence);
        */
        List<String> sentencePieces = new ArrayList<>();
        if (tokenizerModel.getAddedTokenPattern() != null){
            //String[] pieces = TokenizerModel.split(model.getAddedTokenPattern(), sentence, 0, true);
        }
        return null;
    }

    @Override
    public long[] encode(String sentence) {
        return new long[0];
    }

    @Override
    public String decode(long id) {
        return "";
    }

    @Override
    public TokenizerModel getModel() {
        return null;
    }
}
