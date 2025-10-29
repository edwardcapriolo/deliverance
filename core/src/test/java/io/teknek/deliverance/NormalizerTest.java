package io.teknek.deliverance;



import io.teknek.deliverance.tokenizer.Normalizer;
import io.teknek.deliverance.tokenizer.NormalizerItem;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Collections;

public class NormalizerTest {

    @Test
    void basicNormalizer(){
        Normalizer normal = new Normalizer("", Collections.emptyList());
        Assertions.assertEquals("This is co◌̈l", normal.normalize("This is co◌̈l"));
    }


    @Test
    void withAnItem(){
        NormalizerItem i = new NormalizerItem("NFKC","", null, null);
        Normalizer normal = new Normalizer("", Collections.singletonList(i));
        Assertions.assertEquals("This is co◌̈l", normal.normalize("This is co◌̈l"));
    }

}
