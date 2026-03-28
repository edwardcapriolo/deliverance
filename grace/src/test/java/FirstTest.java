import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.concurrent.ExecutionException;

import static org.junit.jupiter.api.Assertions.assertEquals;

/*
from transformers import AutoTokenizer
model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
*/
public class FirstTest {

    @Test
    void startSomewhere(){
        PreTrainedTokenizer autoTokenizer = AutoTokenizer.fromPretrained(
                new AutoTokenizer.OwnerNameOrPath(new AutoTokenizer.OwnerName("Qwen", "Qwen2.5-7B-Instruct")),
                null, null, null);
        {
            //>>> tokenizer.vocab_size
            //151643
            assertEquals(151643, autoTokenizer.getVocabSize());

        }
        {
            //>>> tokenizer.get_added_vocab()
            //{'<|endoftext|>': 151643, '<|im_start|>': 151644, '<|im_end|>': 151645, '<|object_ref_start|>': 151646, '<|object_ref_end|>': 151647, '<|box_start|>': 151648, '<|box_end|>': 151649, '<|quad_start|>': 151650, '<|quad_end|>': 151651, '<|vision_start|>': 151652, '<|vision_end|>': 151653, '<|vision_pad|>': 151654, '<|image_pad|>': 151655, '<|video_pad|>': 151656, '<tool_call>': 151657, '</tool_call>': 151658, '<|fim_prefix|>': 151659, '<|fim_middle|>': 151660, '<|fim_suffix|>': 151661, '<|fim_pad|>': 151662, '<|repo_name|>': 151663, '<|file_sep|>': 151664}
        }
        {
            //>>> tokenizer.all_special_ids
            //    [151645, 151643, 151644, 151646, 151647, 151648, 151649, 151650, 151651, 151652, 151653, 151654, 151655, 151656]
            
        }

    }
}
