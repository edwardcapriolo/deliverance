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
            Response r = m.generate(UUID.randomUUID(), ctx, new GeneratorParameters().withSeed(43).withMaxTokens(15),
                    new DoNothingGenerateEvent());
            Assertions.assertEquals("1 is the next in the sequence. 2 is the next in the", r.responseText);
        }
    }
}
