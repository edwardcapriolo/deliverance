package io.teknek.deliverance.model.llama;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class LlamaTokenizerTest {

    LlamaTokenizer tokenizer = new LlamaTokenizer(Path.of("src/test/resources/tinylama_model_dir"));

    @Test
    void decodeAsciiChar(){
        assertEquals( 100L, tokenizer.encodeCharacterAsToken((byte) Character.valueOf('a').charValue()));
        assertEquals(Optional.of('a'), tokenizer.maybeDecodeTokenAsCharacter(100));
    }

    @Test
    void decodeNonAsciiChar(){
        assertEquals(Optional.empty(), tokenizer.maybeDecodeTokenAsCharacter(999));
    }
}
