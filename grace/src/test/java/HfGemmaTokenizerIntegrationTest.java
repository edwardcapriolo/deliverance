/**
 * Port of Hugging Face GemmaTokenizationTest integration golden.
 *
 * Source: /ai-code/transformers/tests/models/gemma/test_tokenization_gemma.py
 */
public class HfGemmaTokenizerIntegrationTest implements HfTokenizerIntegrationContract {
    @Override
    public io.teknek.deliverance.grace.AutoTokenizer.OwnerName tokenizerOwnerName() {
        return new io.teknek.deliverance.grace.AutoTokenizer.OwnerName("hf-internal-testing", "dummy-gemma");
    }

    @Override
    public String integrationInputString() {
        return HfTokenizerFixtures.HF_COMMON_INPUT;
    }

    @Override
    public String[] integrationExpectedTokens() {
        return new String[]{"This", "▁is", "▁a", "▁test", "▁😊", "\n", "I", "▁was", "▁born", "▁in", "▁", "9", "2", "0", "0", "0", ",", "▁and", "▁this", "▁is", "▁fals", "é", ".", "\n", "生活的", "真", "谛", "是", "\n", "Hi", "▁▁", "Hello", "\n", "Hi", "▁▁▁", "Hello", "\n\n", "▁", "\n", "▁▁", "\n", "▁Hello", "\n", "<", "s", ">", "\n", "hi", "<", "s", ">", "there", "\n", "The", "▁following", "▁string", "▁should", "▁be", "▁properly", "▁encoded", ":", "▁Hello", ".", "\n", "But", "▁i", "rd", "▁and", "▁ปี", "▁▁▁", "ird", "▁▁▁", "ด", "\n", "Hey", "▁how", "▁are", "▁you", "▁doing"};
    }

    @Override
    public int[] integrationExpectedTokenIds() {
        return new int[]{1596, 603, 476, 2121, 44416, 108, 235285, 729, 7565, 575, 235248, 235315, 235284, 235276, 235276, 235276, 235269, 578, 736, 603, 40751, 235335, 235265, 108, 122182, 235710, 245467, 235427, 108, 2151, 139, 4521, 108, 2151, 140, 4521, 109, 235248, 108, 139, 108, 25957, 108, 235322, 235256, 235313, 108, 544, 235322, 235256, 235313, 11048, 108, 651, 2412, 2067, 1412, 614, 10338, 49748, 235292, 25957, 235265, 108, 1860, 496, 1924, 578, 73208, 140, 5650, 140, 235732, 108, 6750, 1368, 708, 692, 3900};
    }

    @Override
    public String integrationExpectedDecodedText() {
        return HfTokenizerFixtures.HF_COMMON_INPUT;
    }
}
