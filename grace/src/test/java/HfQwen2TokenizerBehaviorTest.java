import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.PreTrainedTokenizer;

public class HfQwen2TokenizerBehaviorTest implements HfTokenizerBehaviorContract {
    @Override
    public PreTrainedTokenizer tokenizer() {
        return AutoTokenizer.fromPretrained(new AutoTokenizer.OwnerNameOrPath(
                new AutoTokenizer.OwnerName("Qwen", "Qwen2.5-7B-Instruct")));
    }

    @Override
    public int padTokenId() {
        return 151643;
    }

    @Override
    public int[] sampleTokenIds() {
        return new int[]{1986, 374, 264, 1273};
    }
}
