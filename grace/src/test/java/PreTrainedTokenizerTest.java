import io.teknek.deliverance.grace.PreTrainedTokenizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PreTrainedTokenizerTest {
    @Test
    void punctuationTest(){
       assertFalse(PreTrainedTokenizer.isPunctuation('a'));
       assertTrue(PreTrainedTokenizer.isPunctuation(','));
    }

    @Test
    void controlTest(){
        assertFalse(PreTrainedTokenizer.isControl('a'));
        assertFalse(PreTrainedTokenizer.isControl('\t'));
    }

    @Test
    void punctAtEndOfWordTest(){
        String s = "hello,";
        assertTrue(PreTrainedTokenizer.isEndOfWord(s));
        assertFalse(PreTrainedTokenizer.isEndOfWord("hello"));
    }

    @Test
    void punctAtBeginOfWordTest(){
        String s = ",hello";
        assertTrue(PreTrainedTokenizer.isStartOfWord(s));
        assertFalse(PreTrainedTokenizer.isStartOfWord("hello"));
    }
}
