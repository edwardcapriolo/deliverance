/**
 * Port of Hugging Face Qwen2TokenizationTest integration golden.
 *
 * Source: /ai-code/transformers/tests/models/qwen2/test_tokenization_qwen2.py
 */
public class HfQwen2TokenizerIntegrationTest implements HfTokenizerIntegrationContract {
    @Override
    public io.teknek.deliverance.grace.AutoTokenizer.OwnerName tokenizerOwnerName() {
        return new io.teknek.deliverance.grace.AutoTokenizer.OwnerName("Qwen", "Qwen2.5-7B-Instruct");
    }

    @Override
    public String integrationInputString() {
        return HfTokenizerFixtures.HF_COMMON_INPUT;
    }

    @Override
    public String[] integrationExpectedTokens() {
        return new String[]{"This", "Ġis", "Ġa", "Ġtest", "ĠðŁĺ", "Ĭ", "Ċ", "I", "Ġwas", "Ġborn", "Ġin", "Ġ", "9", "2", "0", "0", "0", ",", "Ġand", "Ġthis", "Ġis", "Ġfals", "Ã©", ".Ċ", "çĶŁæ´»çļĦ", "çľŁ", "è°Ľ", "æĺ¯", "Ċ", "Hi", "Ġ", "ĠHello", "Ċ", "Hi", "ĠĠ", "ĠHello", "ĊĊ", "ĠĊĠĠĊ", "ĠHello", "Ċ", "<s", ">Ċ", "hi", "<s", ">", "there", "Ċ", "The", "Ġfollowing", "Ġstring", "Ġshould", "Ġbe", "Ġproperly", "Ġencoded", ":", "ĠHello", ".Ċ", "But", "Ġ", "ird", "Ġand", "Ġ", "à¸Ľ", "à¸µ", "ĠĠ", "Ġ", "ird", "ĠĠ", "Ġ", "à¸Ķ", "Ċ", "Hey", "Ġhow", "Ġare", "Ġyou", "Ġdoing"};
    }

    @Override
    public int[] integrationExpectedTokenIds() {
        return new int[]{1986, 374, 264, 1273, 26525, 232, 198, 40, 572, 9223, 304, 220, 24, 17, 15, 15, 15, 11, 323, 419, 374, 31932, 963, 624, 105301, 88051, 116109, 20412, 198, 13048, 220, 21927, 198, 13048, 256, 21927, 271, 48426, 21927, 198, 44047, 397, 6023, 44047, 29, 18532, 198, 785, 2701, 914, 1265, 387, 10277, 20498, 25, 21927, 624, 3983, 220, 2603, 323, 220, 54684, 28319, 256, 220, 2603, 256, 220, 37033, 198, 18665, 1246, 525, 498, 3730};
    }

    @Override
    public String integrationExpectedDecodedText() {
        return HfTokenizerFixtures.HF_COMMON_INPUT;
    }
}
