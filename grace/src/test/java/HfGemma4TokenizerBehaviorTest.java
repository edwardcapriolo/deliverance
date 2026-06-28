import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Assumptions;

public class HfGemma4TokenizerBehaviorTest implements HfTokenizerBehaviorContract {
    @Override
    public PreTrainedTokenizer tokenizer() {
        ModelFetcher fetch = new ModelFetcher("edward", "gemma-4-E2B-it-JQ4");
        Assumptions.assumeTrue(fetch.pathForModel().toFile().isDirectory(),
                "Quantized Gemma4 cache is not present: " + fetch.pathForModel());
        return AutoTokenizer.fromPretrained(new AutoTokenizer.OwnerNameOrPath(
                new AutoTokenizer.OwnerName("edward", "gemma-4-E2B-it-JQ4")));
    }

    @Override
    public int padTokenId() {
        return 0;
    }

    @Override
    public int[] sampleTokenIds() {
        return new int[]{105, 2364, 107, 3689};
    }
}
