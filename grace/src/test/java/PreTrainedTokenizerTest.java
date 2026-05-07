import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PreTrainedTokenizerTest {
    @TempDir
    Path tempDir;

    @Test
    void punctuationTest(){
       assertFalse(PreTrainedTokenizer.isPunctuation('a'));
       assertTrue(PreTrainedTokenizer.isPunctuation(','));
    }

    @Test
    void controlTest(){
        assertFalse(PreTrainedTokenizer.isControl('a'));
        assertFalse(PreTrainedTokenizer.isControl('\t'));
    }

    @Test
    void punctAtEndOfWordTest(){
        String s = "hello,";
        assertTrue(PreTrainedTokenizer.isEndOfWord(s));
        assertFalse(PreTrainedTokenizer.isEndOfWord("hello"));
    }

    @Test
    void punctAtBeginOfWordTest(){
        String s = ",hello";
        assertTrue(PreTrainedTokenizer.isStartOfWord(s));
        assertFalse(PreTrainedTokenizer.isStartOfWord("hello"));
    }

    @Test
    void loadsChatTemplateFromJinjaFile() throws Exception {
        Files.writeString(tempDir.resolve("tokenizer.json"), """
                {
                  "model": {
                    "type": "BPE",
                    "unk_token": "<unk>",
                    "vocab": {
                      "<unk>": 0,
                      "hello": 1
                    },
                    "merges": []
                  },
                  "added_tokens": []
                }
                """);
        Files.writeString(tempDir.resolve("tokenizer_config.json"), """
                {
                  "tokenizer_class": "GemmaTokenizer",
                  "eos_token": "<eos>",
                  "bos_token": "<bos>"
                }
                """);
        Files.writeString(tempDir.resolve("chat_template.jinja"), "{{ messages }}");

        PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(tempDir);
        assertEquals("{{ messages }}", tokenizer.chatTemplate().orElseThrow());
    }

}
