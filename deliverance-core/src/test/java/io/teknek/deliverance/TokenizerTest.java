package io.teknek.deliverance;

import io.teknek.deliverance.tokenizer.Tokenizer;
import io.teknek.deliverance.tokenizer.TokenizerModel;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;

public class TokenizerTest {

    @Test
    void readFile(){
        TokenizerModel model = TokenizerModel.load(new File("src/test/resources/tinylama_tok.json"));
        Assertions.assertEquals("BPE", model.type);
    }

}
