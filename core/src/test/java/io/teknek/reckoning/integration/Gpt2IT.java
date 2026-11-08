package io.teknek.reckoning.integration;

import io.teknek.reckoning.generator.GeneratorParameters;
import io.teknek.reckoning.generator.Response;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.AutoModelForCausaLm;
import io.teknek.reckoning.model.DoNothingGenerateEvent;
import io.teknek.reckoning.safetensors.fetch.ModelFetcher;
import io.teknek.reckoning.safetensors.prompt.PromptContext;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertFalse;

@Tag("large-model")
public class Gpt2IT {
    @Test
    void answersMichaelJordanPrompt() {
        ModelFetcher fetcher = new ModelFetcher("openai-community", "gpt2-large");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher)
                .buildLocalTransformerModel()) {
            // GPT-2 is a base completion model and does not provide chat prompt support.
            Response response = model.generate(UUID.randomUUID(),
                    PromptContext.of("Who is Micheal Jordan?"),
                    new GeneratorParameters()
                            .withTemperature(0.0f)
                            .withMaxTokens(12)
                            .withTimeout(Duration.ofSeconds(60)),
                    new DoNothingGenerateEvent());
            System.out.println(response.responseText);
            assertFalse(response.responseText.isBlank());
            Assertions.assertEquals("\n\nMicheal Jordan is a former professional basketball player", response.responseText);
        }
    }
}
