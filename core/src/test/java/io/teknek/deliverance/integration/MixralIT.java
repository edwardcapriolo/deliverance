package io.teknek.deliverance.integration;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MixralIT {
    @Test
    public void chat(){
        ModelFetcher fetch = new ModelFetcher("tjake", "Mixtral-8x7B-Instruct-v0.1-JQ4");

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).build()) {
            String prompt = "Who is Edward Capriolo?";
            PromptSupport.Builder g = model.promptSupport().get().builder()
                    .addUserMessage(prompt);
            var uuid = UUID.randomUUID();

            Response response = model.generate(uuid, g.build(), new GeneratorParameters()
                            .withTemperature(0.5f).withNtokens(500).withMaxTokens(150),
                    new GenerateEvent() {
                        @Override
                        public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                            System.out.println(nextCleaned);
                        }
                    });
            assertEquals("", response.responseText);
        }
    }
}
