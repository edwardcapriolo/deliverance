package io.teknek.deliverance.model;

import io.teknek.deliverance.integration.Gemma2Suite;
import io.teknek.deliverance.model.gemma2.GemmaTokenizer;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ChoiceEncodedTest {

    @Test
    void buildEncoded(){
        AbstractModel m = Gemma2Suite.getOrCreate();
        ChoiceEncoded ci = new ChoiceEncoded(Arrays.asList("Giants", "Jets"), m);
        assertEquals( GemmaTokenizer.class, m.tokenizer.getClass());
        GemmaTokenizer gemmaT = (GemmaTokenizer) m.tokenizer;
        //Important: just because a model has the complete token in the vocabulary it might not be an optioa after
        //forward pass
        assertEquals(2, ci.getEncoded().size());
        assertEquals(List.of(218954L), ci.getEncoded().get("Giants"));
        assertEquals("Giants", m.tokenizer.decode(218954));
    }

    @Test
    void longerName(){
        AbstractModel m = Gemma2Suite.getOrCreate();
        ChoiceEncoded ci = new ChoiceEncoded(List.of("New York football Giants"), m);
        assertEquals(GemmaTokenizer.class, m.tokenizer.getClass());
        GemmaTokenizer gemmaT = (GemmaTokenizer) m.tokenizer;
        assertEquals(1, ci.getEncoded().size());//
        assertEquals(Arrays.asList(2441L, 3459L, 9715L, 54795L),
                ci.getEncoded().get("New York football Giants"));
    }

    @Test
    void prefixMatchingUsesEncodedTokenIds(){
        AbstractModel m = Gemma2Suite.getOrCreate();
        ChoiceEncoded ci = new ChoiceEncoded(List.of("Giants"), m);
        assertEquals(true, ci.anyStartsWith(List.of(218954)));
        assertEquals(false, ci.anyStartsWith(List.of(2441)));
    }
}
