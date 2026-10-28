import io.teknek.deliverance.grace.EncodeOptions;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.grace.TokenIds;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** Port of Hugging Face XLMRobertaTokenizationTest integration golden data. */
public class HfXlmRobertaTokenizerIntegrationTest implements HfTokenizerIntegrationContract {
    @Override
    public io.teknek.deliverance.grace.AutoTokenizer.OwnerName tokenizerOwnerName() {
        return new io.teknek.deliverance.grace.AutoTokenizer.OwnerName(
                "papluca", "xlm-roberta-base-language-detection");
    }

    @Override
    public String integrationInputString() {
        return HfTokenizerFixtures.HF_COMMON_INPUT;
    }

    @Override
    public String[] integrationExpectedTokens() {
        return new String[]{"▁This", "▁is", "▁a", "▁test", "▁", "😊", "▁I", "▁was", "▁born", "▁in", "▁9", "2000", ",", "▁and", "▁this", "▁is", "▁fals", "é", ".", "▁", "生活的", "真", "谛", "是", "▁Hi", "▁Hello", "▁Hi", "▁Hello", "▁Hello", "<s>", "▁hi", "<s>", "▁there", "▁The", "▁following", "▁string", "▁should", "▁be", "▁properly", "▁en", "code", "d", ":", "▁Hello", ".", "▁But", "▁ir", "d", "▁and", "▁ปี", "▁ir", "d", "▁ด", "▁Hey", "▁how", "▁are", "▁you", "▁doing"};
    }

    @Override
    public int[] integrationExpectedTokenIds() {
        return new int[]{3293, 83, 10, 3034, 6, 82803, 87, 509, 103122, 23, 483, 13821, 4, 136, 903, 83, 84047, 446, 5, 6, 62668, 5364, 245875, 354, 2673, 35378, 2673, 35378, 35378, 0, 1274, 0, 2685, 581, 25632, 79315, 5608, 186, 155965, 22, 40899, 71, 12, 35378, 5, 4966, 193, 71, 136, 10249, 193, 71, 48229, 28240, 3642, 621, 398, 20594};
    }

    @Test
    public void encodedIdsMatchHfGolden() {
        PreTrainedTokenizer tokenizer = tokenizer();
        assertEquals(Arrays.toString(integrationExpectedTokenIds()),
                Arrays.toString(tokenizer.encode(integrationInputString(), EncodeOptions.defaults().withoutSpecialTokens()).inputIds()));
    }

    @Override
    public String integrationExpectedDecodedText() {
        return "This is a test 😊 I was born in 92000, and this is falsé. 生活的真谛是 Hi Hello Hi Hello Hello<s> hi<s> there The following string should be properly encoded: Hello. But ird and ปี ird ด Hey how are you doing";
    }
}
