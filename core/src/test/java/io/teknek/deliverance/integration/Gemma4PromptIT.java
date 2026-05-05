package io.teknek.deliverance.integration;


import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.model.gemma4.Gemma4ResponseParser;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class Gemma4PromptIT {
    @Test

    public void chatWithThinking() {
        AbstractModel model = Gemma4Suite.getOrCreate();
        PromptSupport.Builder builder = model.promptSupport().get().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                .addSystemMessage("You are a concise assistant.")
                .addUserMessage("What is the capital of New York?");
        Response response = model.generate(
                UUID.randomUUID(),
                builder.build(),
                new GeneratorParameters().withTemperature(0.0f).withMaxTokens(256),
                new DoNothingGenerateEvent()
        );
        Gemma4ResponseParser.Parsed parsed = Gemma4ResponseParser.parse(
                response.responseTextWithSpecialTokens,
                response.responseText
        );
        assertTrue(parsed.content().contains("Albany"));
    }
}
