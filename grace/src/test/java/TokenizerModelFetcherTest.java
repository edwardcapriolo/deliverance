import io.teknek.deliverance.grace.TokenizerModelFetcher;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class TokenizerModelFetcherTest {
    @Test
    void tokenizerFetcherUsesSameModelDirectoryAsModelFetcher() {
        ModelFetcher modelFetcher = new ModelFetcher("google", "gemma-4-E2B-it");
        TokenizerModelFetcher tokenizerFetcher = new TokenizerModelFetcher("google", "gemma-4-E2B-it");

        Assertions.assertEquals(modelFetcher.pathForModel(), tokenizerFetcher.pathForModel());
    }
}
