package io.teknek.deliverance.tokenizer;


import java.util.Arrays;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.fetch.ModelFetcher;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class LlamaTokenizerTest {

    @Test
    void endcodingDecodingTest(){
        String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8);
        List<String> tokens = m.getTokenizer().tokenize("show me the money!");
        assertEquals(Arrays.asList("show me the money!"), tokens);
        long [] encode = m.getTokenizer().encode("show me!");
        Assertions.assertArrayEquals(new long[] {4294,35,1004,29991},encode );
        assertEquals("show", m.getTokenizer().decode(4294));
        assertEquals("me", m.getTokenizer().decode(1004));
        assertEquals("!", m.getTokenizer().decode(29991));
    }
}
