import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.grace.TokenIds;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Port of the integration portion of Hugging Face's TokenizerTesterMixin.
 *
 * Source: /ai-code/transformers/tests/test_tokenization_common.py
 * Methods ported: _run_integration_checks / test_integration.
 */
public interface HfTokenizerIntegrationContract {

    AutoTokenizer.OwnerName tokenizerOwnerName();

    String integrationInputString();

    String[] integrationExpectedTokens();

    int[] integrationExpectedTokenIds();

    String integrationExpectedDecodedText();

    @Test
    default void tokenizedTokensMatchHfGolden() {
        PreTrainedTokenizer tokenizer = tokenizer();

        assertEquals(Arrays.toString(integrationExpectedTokens()),
                Arrays.toString(tokenizer.tokenize(integrationInputString()).getInputs()));
    }

    @Test
    default void encodedIdsMatchHfGolden() {
        PreTrainedTokenizer tokenizer = tokenizer();

        assertEquals(Arrays.toString(integrationExpectedTokenIds()),
                Arrays.toString(tokenizer.encode(integrationInputString()).inputIds()));
    }

    @Test
    default void decodedTextMatchesHfGolden() {
        PreTrainedTokenizer tokenizer = tokenizer();

        assertEquals(integrationExpectedDecodedText(),
                tokenizer.decode(new TokenIds(integrationExpectedTokenIds()), false, false, false, false));
    }

    default PreTrainedTokenizer tokenizer() {
        return AutoTokenizer.fromPretrained(new AutoTokenizer.OwnerNameOrPath(tokenizerOwnerName()));
    }
}
