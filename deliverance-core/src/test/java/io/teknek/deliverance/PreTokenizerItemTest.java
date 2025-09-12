package io.teknek.deliverance;

import io.teknek.deliverance.tokenizer.PatternModel;
import io.teknek.deliverance.tokenizer.PretokenizerItem;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class PreTokenizerItemTest {

    @Test
    void assertDigits(){
        PretokenizerItem i = new PretokenizerItem(PretokenizerItem.DIGITS, null, null, null, null, null, null, null  );
        assertEquals(List.of("55", " is better then ", "55", ".", "01", " and ", "55", ",",  "000", ",", "000", " yet -", "33", ".", "0", " is a quandry "),
                i.pretokenize("55 is better then 55.01 and 55,000,000 yet -33.0 is a quandry "));
    }

    @Test
    void assertSplit(){
        PretokenizerItem i = new PretokenizerItem(PretokenizerItem.SPLIT, new PatternModel("\\s"), null, null, null, null, null, null  );
        assertEquals(List.of("lets", " ", "go", " ", "home"),
                i.pretokenize("lets go home"));
    }

}
