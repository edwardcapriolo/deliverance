package io.teknek.deliverance;

import io.teknek.deliverance.tokenizer.Tokenizer;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.List;

public class QwenTokenizerTest {

    /*
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
except:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    print(tokenizer.encode(" hello sir. How are you Edward?", add_special_tokens=False))
    [23811, 27048, 13, 2585, 525, 498, 21891, 30]
     */
    @Disabled
    void encodeTest(){
        Tokenizer t = null;
        long[] result = t.encode(" hello sir. How are you Edward?");
        Assertions.assertArrayEquals(new long []{ 23811, 27048, 13, 2585, 525, 498, 21891, 30}, result);
    }
}
