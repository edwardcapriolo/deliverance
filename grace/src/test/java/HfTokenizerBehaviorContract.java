import io.teknek.deliverance.grace.BatchEncoding;
import io.teknek.deliverance.grace.Encoding;
import io.teknek.deliverance.grace.PaddingOptions;
import io.teknek.deliverance.grace.PaddingSide;
import io.teknek.deliverance.grace.PaddingStrategy;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import io.teknek.deliverance.grace.TokenIds;
import io.teknek.deliverance.grace.TruncationOptions;
import io.teknek.deliverance.grace.TruncationSide;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Additional Hugging Face tokenizer common-behavior contract.
 *
 * <p>Ports behavior categories from /ai-code/transformers/tests/test_tokenization_common.py around empty input,
 * truncation, padding, and batch padding. Model-specific classes provide a tokenizer and stable sample token ids.</p>
 */
public interface HfTokenizerBehaviorContract {

    PreTrainedTokenizer tokenizer();

    int padTokenId();

    int[] sampleTokenIds();

    @Test
    default void emptyInputEncodesToEmptySequence() {
        Encoding encoding = tokenizer().encode("");

        assertArrayEquals(new int[0], encoding.inputIds());
        assertArrayEquals(new int[0], encoding.attentionMask());
    }

    @Test
    default void truncatesFromRightAndLeft() {
        int[] sample = sampleTokenIds();

        Encoding right = tokenizer().truncate(new TokenIds(sample), TruncationOptions.maxLength(2));
        Encoding left = tokenizer().truncate(new TokenIds(sample), TruncationOptions.maxLength(2).withSide(TruncationSide.LEFT));

        assertArrayEquals(new int[]{sample[0], sample[1]}, right.inputIds());
        assertArrayEquals(new int[]{sample[sample.length - 2], sample[sample.length - 1]}, left.inputIds());
    }

    @Test
    default void padsSingleSequenceOnConfiguredSide() {
        int[] sample = sampleTokenIds();

        Encoding padded = tokenizer().pad(new TokenIds(new int[]{sample[0], sample[1]}), PaddingOptions.maxLength(4));

        if (tokenizer().paddingSide() == PaddingSide.LEFT) {
            assertArrayEquals(new int[]{padTokenId(), padTokenId(), sample[0], sample[1]}, padded.inputIds());
            assertArrayEquals(new int[]{0, 0, 1, 1}, padded.attentionMask());
        } else {
            assertArrayEquals(new int[]{sample[0], sample[1], padTokenId(), padTokenId()}, padded.inputIds());
            assertArrayEquals(new int[]{1, 1, 0, 0}, padded.attentionMask());
        }
    }

    @Test
    default void padsBatchToLongestSequence() {
        int[] sample = sampleTokenIds();

        BatchEncoding batch = tokenizer().pad(List.of(
                new TokenIds(new int[]{sample[0], sample[1], sample[2]}),
                new TokenIds(new int[]{sample[0]})
        ), new PaddingOptions(PaddingStrategy.LONGEST, null, null, null));

        assertEquals(2, batch.encodings().size());
        assertEquals(3, batch.encodings().get(0).inputIds().length);
        assertEquals(3, batch.encodings().get(1).inputIds().length);
        if (tokenizer().paddingSide() == PaddingSide.LEFT) {
            assertArrayEquals(new int[]{padTokenId(), padTokenId(), sample[0]}, batch.encodings().get(1).inputIds());
        } else {
            assertArrayEquals(new int[]{sample[0], padTokenId(), padTokenId()}, batch.encodings().get(1).inputIds());
        }
    }
}
