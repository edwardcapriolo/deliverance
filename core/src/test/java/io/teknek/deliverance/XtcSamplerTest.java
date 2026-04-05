package io.teknek.deliverance;

import io.teknek.deliverance.model.ExcludeTopChoicePicker;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Optional;
import java.util.Random;

public class XtcSamplerTest {

    @Test
    public void samplerTest(){
        ExcludeTopChoicePicker x = new ExcludeTopChoicePicker(null, null, 0.1f, 0.0001f, new Random());
        Assertions.assertEquals(Optional.empty(), x.process());
    }
}
