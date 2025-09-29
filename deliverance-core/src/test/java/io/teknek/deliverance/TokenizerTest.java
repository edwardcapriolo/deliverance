package io.teknek.deliverance;

import io.teknek.deliverance.tokenizer.TokenizerModel;
import org.junit.jupiter.api.Test;

import java.io.File;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class TokenizerTest {

    @Test
    void readFile() {
        TokenizerModel model = TokenizerModel.load(new File("src/test/resources/tinylama_tok.json"),
                new File("src/test/resources/tinylama_tok_config.json"));
        assertEquals("BPE", model.type);
        assertNotNull(model.getNormalizer());
        assertEquals("<s>", model.getBosToken());
        //{<unk>=0, <s>=1, </s>=2}
        assertEquals(3, model.getAddedTokens().size());
        assertEquals(1, model.getAddedTokens().get("<s>"));
        assertEquals("default", model.getPromptTemplates().get().keySet().stream().iterator().next());
    }
}
