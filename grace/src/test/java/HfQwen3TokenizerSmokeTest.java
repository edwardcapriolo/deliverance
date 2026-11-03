import io.teknek.reckoning.grace.AutoTokenizer;
import io.teknek.reckoning.grace.Encoding;
import io.teknek.reckoning.grace.PreTrainedTokenizer;
import io.teknek.reckoning.grace.TokenIds;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class HfQwen3TokenizerSmokeTest {

    @Test
    public void qwen3TokenizerLoadsAndEncodesThinkingMarkers() {
        PreTrainedTokenizer tokenizer = AutoTokenizer.fromPretrained(new AutoTokenizer.OwnerNameOrPath(
                new AutoTokenizer.OwnerName("Qwen", "Qwen3-0.6B")));

        assertTrue(tokenizer.chatTemplate().isPresent());
        Encoding think = tokenizer.encode("<think>\nreason\n</think>\nanswer");
        String decoded = tokenizer.decode(new TokenIds(think.inputIds()), false, false, false, false);

        assertEquals("<think>\nreason\n</think>\nanswer", decoded);
        assertFalse(think.inputIds().length == 0);
    }
}
