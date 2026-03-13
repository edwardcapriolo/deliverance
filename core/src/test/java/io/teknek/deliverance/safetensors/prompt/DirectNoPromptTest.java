package io.teknek.deliverance.safetensors.prompt;

import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.UUID;

public class DirectNoPromptTest {

    @Test
    public void noPromptContext() throws IOException {
        ModelFetcher fetch = new ModelFetcher("lidoreliya13", "microlama-lidor-finetuned");
        try (AbstractModel m = AutoModelForCausaLm.fromPretrained(fetch)) {
            String prompt = "What comes next in the sequence? 1, 2 ";
            PromptContext ctx = PromptContext.of(prompt);
            Response r = m.generate(UUID.randomUUID(), ctx, new GeneratorParameters().withSeed(43), new DoNothingGenerateEvent());
            Assertions.assertEquals("2,33,44,55,66,77,88,99,1010,1111,1212,1313,1414,1515,1616,1717,1818,1919,2020,2121,2222,2322,2422,2522,2622,2722,2822,2922,3022,3122,3222,3322,3422,3522,3622,3722,382", r.responseText);
        }
    }
}
