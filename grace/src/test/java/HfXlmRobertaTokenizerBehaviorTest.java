import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.BatchEncoding;
import io.teknek.deliverance.grace.EncodeOptions;
import io.teknek.deliverance.grace.PaddingOptions;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class HfXlmRobertaTokenizerBehaviorTest {
    private PreTrainedTokenizer tokenizer() {
        return AutoTokenizer.fromPretrained(new AutoTokenizer.OwnerNameOrPath(
                new AutoTokenizer.OwnerName("papluca", "xlm-roberta-base-language-detection")));
    }

    @Test
    public void addsXlmRobertaSpecialTokens() {
        var encoding = tokenizer().encode("This is English.");
        assertEquals(0, encoding.inputIds()[0]);
        assertEquals(2, encoding.inputIds()[encoding.inputIds().length - 1]);
        assertEquals(1, encoding.specialTokensMask()[0]);
        assertEquals(1, encoding.specialTokensMask()[encoding.specialTokensMask().length - 1]);
    }

    @Test
    public void padsBatchWithPadTokenAndAttentionMask() {
        BatchEncoding batch = tokenizer().encode(List.of("Hello", "Hello world"),
                EncodeOptions.defaults().withPadding(PaddingOptions.longest()));
        int[][] masks = batch.encodings().stream().map(e -> e.attentionMask()).toArray(int[][]::new);
        assertEquals(1, masks[0][0]);
        assertEquals(0, masks[0][masks[0].length - 1]);
        assertTrue(masks[1][masks[1].length - 1] == 1);
    }

    @Test
    public void truncatesAfterAddingSpecialTokens() {
        var encoding = tokenizer().encode("This is a deliberately long sentence for truncation.",
                EncodeOptions.defaults().withTruncation(io.teknek.deliverance.grace.TruncationOptions.maxLength(6)));
        assertEquals(6, encoding.inputIds().length);
        assertEquals(0, encoding.inputIds()[0]);
        assertEquals(2, encoding.inputIds()[5]);
    }
}
