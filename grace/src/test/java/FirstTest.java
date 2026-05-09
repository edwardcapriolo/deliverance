import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/*
model_id = 'google/gemma-4-E2B-it'
tokenizer = AutoTokenizer.from_pretrained(model_id)

from transformers import AutoTokenizer
model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
*/
public class FirstTest {

    @Test
    void qwenMetadataMatchesCachedTokenizerOutput() {
        Path cachedFixture = Path.of("target", "test-classes", "Qwen_Qwen2.5-7B-Instruct");
        Assumptions.assumeTrue(Files.exists(cachedFixture.resolve("tokenizer.json")));
        Assumptions.assumeTrue(Files.exists(cachedFixture.resolve("tokenizer_config.json")));

        PreTrainedTokenizer autoTokenizer = AutoTokenizer.fromPretrained(cachedFixture);

        // >>> tokenizer.vocab_size
        // 151643
        assertEquals(151643, autoTokenizer.getVocabSize());

        // >>> tokenizer.get_added_vocab()
        // {'<|endoftext|>': 151643, '<|im_start|>': 151644, '<|im_end|>': 151645, ... '<|file_sep|>': 151664}
        assertEquals(22, autoTokenizer.getAddedVocab().size());
        assertEquals(151643, autoTokenizer.getAddedVocab().get("<|endoftext|>"));
        assertEquals(151664, autoTokenizer.getAddedVocab().get("<|file_sep|>"));

        // >>> tokenizer.all_special_ids
        // [151645, 151643, 151644, 151646, 151647, 151648, 151649, 151650, 151651, 151652, 151653, 151654, 151655, 151656]
        List<Integer> expectedSpecialIds = List.of(
                151645, 151643, 151644, 151646, 151647, 151648, 151649,
                151650, 151651, 151652, 151653, 151654, 151655, 151656);
        assertEquals(expectedSpecialIds, autoTokenizer.allSpecialIds());

        // >>> tokenizer.chat_template
        // '{%- if tools %}\n    {{- ...
        assertTrue(autoTokenizer.chatTemplate().orElseThrow().startsWith("{%- if tools %}"));

        // >>> tokenizer.all_special_tokens
        // ['<|im_end|>', '<|endoftext|>', '<|im_start|>', ... '<|video_pad|>']
        List<String> expectedSpecialTokens = List.of(
                "<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|object_ref_start|>",
                "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>",
                "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
                "<|image_pad|>", "<|video_pad|>");
        assertEquals(expectedSpecialTokens, autoTokenizer.allSpecialTokens());

        // >>> tokenizer.special_tokens_map
        // {'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>'}
        Map<String, String> expectedSpecialTokenMap = new LinkedHashMap<>();
        expectedSpecialTokenMap.put("eos_token", "<|im_end|>");
        expectedSpecialTokenMap.put("pad_token", "<|endoftext|>");
        assertEquals(expectedSpecialTokenMap, autoTokenizer.specialTokensMap());

        assertEquals("<|im_start|>", autoTokenizer.tokenize("<|im_start|>").getInputs()[0]);
        assertEquals(151644, autoTokenizer.encode("<|im_start|>").inputIds()[0]);
    }
}
