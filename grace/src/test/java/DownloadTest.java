import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.PreTrainedTokenizer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class DownloadTest {

    @Test
    void downloadTest(){
        PreTrainedTokenizer l = AutoTokenizer.fromPretrained(new AutoTokenizer
                .OwnerNameOrPath(new AutoTokenizer.OwnerName("Qwen", "Qwen2.5-7B-Instruct")));
        assertEquals(151643, l.getVocabSize());
    }
}
