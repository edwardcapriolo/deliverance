import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Golden-style tokenizer test for a locally cached Gemma model.
 *
 * This is intentionally path-based instead of downloading a tokenizer during tests. Once the expected
 * ids are blessed on a dev machine, this gives the grace module a concrete tokenizer regression guard.
 */
public class GemmaTokenizerGoldenTest {
    @Test
    void gemma4PromptGolden() {
        PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(
                new AutoTokenizer.OwnerNameOrPath(new AutoTokenizer.OwnerName("google", "gemma-4-E2B-it")));

        String prompt = """
                <|turn>system
                You are a concise assistant.<turn|>
                <|turn>user
                What is the capital of New York?<turn|>
                <|turn>model
                """;

        int[] actualIds = tokenizer.encode(prompt).inputIds();
        String actualDecoded = tokenizer.decode(new io.teknek.deliverance.grace.TokenIds(actualIds), false, false, false, false);

        /*
python3 - <<'PY'
from transformers import AutoTokenizer

prompt = """<|turn>system
You are a concise assistant.<turn|>
<|turn>user
What is the capital of New York?<turn|>
<|turn>model
"""

tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")

enc = tok(prompt, add_special_tokens=False)
ids = enc["input_ids"]

print("HF_PROMPT_TOKEN_IDS =", ids)
print("HF_TOKEN_COUNT =", len(ids))
print("HF_TOKENS =", tok.convert_ids_to_tokens(ids))
print("HF_DECODED_START")
print(tok.decode(ids, skip_special_tokens=False))
print("HF_DECODED_END")
PY
[transformers] PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
HF_PROMPT_TOKEN_IDS = [105, 9731, 107, 3048, 659, 496, 63510, 16326, 236761, 106, 107, 105, 2364, 107, 3689, 563, 506, 5279, 529, 1799, 3773, 236881, 106, 107, 105, 4368, 107]
HF_TOKEN_COUNT = 27
HF_TOKENS = ['<|turn>', 'system', '\n', 'You', '▁are', '▁a', '▁concise', '▁assistant', '.', '<turn|>', '\n', '<|turn>', 'user', '\n', 'What', '▁is', '▁the', '▁capital', '▁of', '▁New', '▁York', '?', '<turn|>', '\n', '<|turn>', 'model', '\n']
HF_DECODED_START
<|turn>system
You are a concise assistant.<turn|>
<|turn>user
What is the capital of New York?<turn|>
<|turn>model

HF_DECODED_END


         */
        int[] expectedIds = new int[]{
                105, 9731, 107, 3048, 659, 496, 63510, 16326, 236761, 106,
                107, 105, 2364, 107, 3689, 563, 506, 5279, 529, 1799,
                3773, 236881, 106, 107, 105, 4368, 107
        };

        assertEquals(Arrays.toString(expectedIds), Arrays.toString(actualIds));
        assertEquals(prompt, actualDecoded);
    }
}
