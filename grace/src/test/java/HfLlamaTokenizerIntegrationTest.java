/**
 * Port of Hugging Face LlamaTokenizationTest integration golden.
 *
 * Source: /ai-code/transformers/tests/models/llama/test_tokenization_llama.py
 */
public class HfLlamaTokenizerIntegrationTest implements HfTokenizerIntegrationContract {
    @Override
    public io.teknek.deliverance.grace.AutoTokenizer.OwnerName tokenizerOwnerName() {
        return new io.teknek.deliverance.grace.AutoTokenizer.OwnerName("hf-internal-testing", "llama-tokenizer");
    }

    @Override
    public String integrationInputString() {
        return HfTokenizerFixtures.HF_COMMON_INPUT;
    }

    @Override
    public String[] integrationExpectedTokens() {
        return new String[]{"▁This", "▁is", "▁a", "▁test", "▁", "<0xF0>", "<0x9F>", "<0x98>", "<0x8A>", "<0x0A>", "I", "▁was", "▁born", "▁in", "▁", "9", "2", "0", "0", "0", ",", "▁and", "▁this", "▁is", "▁f", "als", "é", ".", "<0x0A>", "生", "活", "的", "真", "<0xE8>", "<0xB0>", "<0x9B>", "是", "<0x0A>", "Hi", "▁", "▁Hello", "<0x0A>", "Hi", "▁▁", "▁Hello", "<0x0A>", "<0x0A>", "▁", "<0x0A>", "▁▁", "<0x0A>", "▁Hello", "<0x0A>", "<s>", "<0x0A>", "hi", "<s>", "there", "<0x0A>", "The", "▁following", "▁string", "▁should", "▁be", "▁properly", "▁encoded", ":", "▁Hello", ".", "<0x0A>", "But", "▁", "ird", "▁and", "▁", "ป", "ี", "▁▁▁", "ird", "▁▁▁", "ด", "<0x0A>", "H", "ey", "▁how", "▁are", "▁you", "▁doing"};
    }

    @Override
    public int[] integrationExpectedTokenIds() {
        return new int[]{910, 338, 263, 1243, 29871, 243, 162, 155, 141, 13, 29902, 471, 6345, 297, 29871, 29929, 29906, 29900, 29900, 29900, 29892, 322, 445, 338, 285, 1338, 29948, 29889, 13, 30486, 31704, 30210, 30848, 235, 179, 158, 30392, 13, 18567, 29871, 15043, 13, 18567, 259, 15043, 13, 13, 29871, 13, 259, 13, 15043, 13, 1, 13, 2918, 1, 12711, 13, 1576, 1494, 1347, 881, 367, 6284, 18511, 29901, 15043, 29889, 13, 6246, 29871, 1823, 322, 29871, 31010, 30691, 1678, 1823, 1678, 30718, 13, 29950, 1032, 920, 526, 366, 2599};
    }

    @Override
    public String integrationExpectedDecodedText() {
        return HfTokenizerFixtures.HF_COMMON_INPUT;
    }
}
