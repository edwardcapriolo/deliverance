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
            Assertions.assertEquals("2,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,", r.responseText);
        }
    }
}
