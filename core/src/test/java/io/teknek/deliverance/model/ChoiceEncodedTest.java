package io.teknek.deliverance.model;

import io.teknek.deliverance.integration.Gemma2Suite;
import io.teknek.deliverance.grace.TokenIds;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class ChoiceEncodedTest {

    @Test
    void buildEncoded(){
        AbstractModel m = Gemma2Suite.getOrCreate();
        ChoiceEncoded ci = new ChoiceEncoded(Arrays.asList("Giants", "Jets"), m);
        //Important: just because a model has the complete token in the vocabulary it might not be an optioa after
        //forward pass
        assertEquals(2, ci.getEncoded().size());
        assertEquals(List.of(218954L), ci.getEncoded().get("Giants"));
        assertEquals("Giants", m.getTokenizer().decode(new TokenIds(218954), false, false, false, false));
    }

    @Test
    void longerName(){
        AbstractModel m = Gemma2Suite.getOrCreate();
        ChoiceEncoded ci = new ChoiceEncoded(List.of("New York football Giants"), m);
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
