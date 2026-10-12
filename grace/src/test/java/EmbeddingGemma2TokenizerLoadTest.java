import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.EncodeOptions;
import io.teknek.deliverance.grace.Encoding;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EmbeddingGemma2TokenizerLoadTest {
    @Test
    @Tag("integration")
    void readsConfiguredBosEosAndPadTokenIds() throws Exception {
        PreTrainedTokenizer tokenizer = loadTokenizer();

        assertEquals(2, tokenizer.bosTokenId().orElseThrow());
        assertEquals(1, tokenizer.eosTokenId().orElseThrow());
        assertEquals(0, tokenizer.padTokenId().orElseThrow());
    }

    @Test
    @Tag("integration")
    void resolvesEmbeddingGemma2ModalityTokenIds() throws Exception {
        PreTrainedTokenizer tokenizer = loadTokenizer();

        assertEquals(258880, tokenizer.tokenToId("<|image|>").orElseThrow());
        assertEquals(258884, tokenizer.tokenToId("<|video|>").orElseThrow());
        assertEquals(258881, tokenizer.tokenToId("<|audio|>").orElseThrow());
    }

    @Test
    @Tag("integration")
    /** Regression test for {@code grace/embeddinggemma2_special_token_bug.md}. */
    void defaultEncodingAddsBosAndEos() throws Exception {
        PreTrainedTokenizer tokenizer = loadTokenizer();
        Encoding encoding = tokenizer.encode("The quick brown fox.");

        assertTrue(encoding.length() > 2);
        assertEquals(encoding.length(), encoding.attentionMask().length);
        assertEquals(2, encoding.inputIds()[0], "EmbeddingGemma 2 must add the BOS token");
        assertEquals(1, encoding.inputIds()[encoding.length() - 1],
                "EmbeddingGemma 2 must add the EOS token");
        assertTrue(java.util.Arrays.stream(encoding.attentionMask()).allMatch(value -> value == 1));
    }

    @Test
    @Tag("integration")
    void encodingCanDisableSpecialTokens() throws Exception {
        PreTrainedTokenizer tokenizer = loadTokenizer();
        Encoding withSpecialTokens = tokenizer.encode("The quick brown fox.");
        Encoding withoutSpecialTokens = tokenizer.encode("The quick brown fox.",
                EncodeOptions.defaults().withoutSpecialTokens());

        assertEquals(withSpecialTokens.length() - 2, withoutSpecialTokens.length());
        assertEquals(2, withSpecialTokens.length() - withoutSpecialTokens.length());
        assertFalse(java.util.Arrays.stream(withoutSpecialTokens.inputIds()).anyMatch(id -> id == 2 || id == 1));
    }

    @Test
    @Tag("integration")
    void embeddingEmptyInputAddsBosAndEos() throws Exception {
        Encoding encoding = loadTokenizer().encode("");

        assertArrayEquals(new int[]{2, 1}, encoding.inputIds());
        assertArrayEquals(new int[]{1, 1}, encoding.attentionMask());
    }

    @Test
    @Tag("integration")
    void embeddingEmptyInputWithoutSpecialTokensIsEmpty() throws Exception {
        Encoding encoding = loadTokenizer().encode("", EncodeOptions.defaults().withoutSpecialTokens());

        assertEquals(0, encoding.length());
        assertEquals(0, encoding.attentionMask().length);
    }

    private static PreTrainedTokenizer loadTokenizer() throws Exception {
        Path cache = Files.createTempDirectory("embeddinggemma2-tokenizer");
        ModelFetcher fetcher = new ModelFetcher("google", "embeddinggemma-2");
        fetcher.setBaseDir(cache);
        File modelDirectory = fetcher.maybeDownload(ModelFetcher.FetchPolicy.TOKENIZER_ONLY);
        assertFalse(Files.exists(modelDirectory.toPath().resolve("model.safetensors")));
        return AutoTokenizer.fromPretrained(modelDirectory.toPath());
    }
}
