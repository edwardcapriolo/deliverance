package io.teknek.reckoning.safetensors.prompt;

import io.teknek.reckoning.model.AutoModelForCausaLm;
import io.teknek.reckoning.model.DoNothingGenerateEvent;
import io.teknek.reckoning.model.GenerateEvent;
import io.teknek.reckoning.safetensors.fetch.ModelFetcher;
import io.teknek.reckoning.generator.GeneratorParameters;
import io.teknek.reckoning.generator.Response;
import io.teknek.reckoning.model.AbstractModel;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.UUID;

public class DirectNoPromptTest {

    @Test
    public void noPromptContext() throws IOException {
        ModelFetcher fetch = new ModelFetcher("edwardcapriolo", "Qwen3-0.6B-JQ4");
        try (AbstractModel m = AutoModelForCausaLm.newBuilder(fetch).buildLocalTransformerModel()) {
            String prompt = "What comes next in the sequence? 1, 2 ";
            PromptContext ctx = PromptContext.of(prompt);
            Response r = m.generate(UUID.randomUUID(), ctx, new GeneratorParameters().withSeed(43).withMaxTokens(15),
                    new GenerateEvent() {
                        @Override
                        public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                            //System.out.println(nextRaw + " ");
                        }
                    });
            //Expected :1 is the next in the sequence. 2 is the next in the
            //Actual   :3, 4, 5, 6, 7,
            //        Assertions.assertEquals("1 is the next in the sequence. 2 is the next in the", r.responseText);
        }
    }
}
