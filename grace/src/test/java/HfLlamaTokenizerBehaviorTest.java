import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.PreTrainedTokenizer;

/** Llama-specific padding behavior used by causal Sentence-Transformers models. */
public class HfLlamaTokenizerBehaviorTest implements HfTokenizerBehaviorContract {
    private PreTrainedTokenizer tokenizer;

    @Override
    public PreTrainedTokenizer tokenizer() {
        if (tokenizer == null) {
            tokenizer = AutoTokenizer.fromPretrained(new AutoTokenizer.OwnerNameOrPath(
                new AutoTokenizer.OwnerName("hf-internal-testing", "llama-tokenizer")));
            tokenizer.withPadToken(tokenizer.eosToken().orElseThrow());
        }
        return tokenizer;
    }

    @Override
    public int padTokenId() {
        return tokenizer().padTokenId().orElseThrow();
    }

    @Override
    public int[] sampleTokenIds() {
        return tokenizer().encode("This is a test", io.teknek.deliverance.grace.EncodeOptions.defaults()
                .withoutSpecialTokens()).inputIds();
    }
}
