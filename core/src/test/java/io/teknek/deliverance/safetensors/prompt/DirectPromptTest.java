package io.teknek.deliverance.safetensors.prompt;

import io.teknek.deliverance.integration.TinyLlamaSuite;
import io.teknek.deliverance.model.*;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;

public class DirectPromptTest {

    @Test
    public void sample() throws IOException {
        AbstractModel m = TinyLlamaSuite.getOrCreate();
        String prompt = "What is the best season to plant avocados?";
        PromptContext ctx;
        {
            PromptSupport ps = m.promptSupport().get();
            ctx = ps.builder().addSystemMessage("You are a chatbot that writes short correct responses.")
                    .addUserMessage(prompt).build();
            String expected = """
                    <|system|>
                    You are a chatbot that writes short correct responses.</s>
                    <|user|>
                    What is the best season to plant avocados?</s>
                    <|assistant|>
                    """;
            assertEquals(expected, ctx.getPrompt());
        }
        {
            PromptSupport ps = m.promptSupport().get();
            ctx = ps.builder().addSystemMessage("You are a chatbot that writes short correct responses.")
                    .addUserMessage(prompt).build();
            String expected = """
                    <|system|>
                    You are a chatbot that writes short correct responses.</s>
                    <|user|>
                    What is the best season to plant avocados?</s>
                    <|assistant|>
                    """;
            assertEquals(expected, ctx.getPrompt());

            m.generate(UUID.randomUUID(), ctx, new GeneratorParameters().withSeed(42).withMaxTokens(50),
                    new DoNothingGenerateEvent());
            assertTrue(TinyLlamaSuite.getBuilder().getMr().meter("tensorcache.dirtyget").getCount() > 100);
        }

    }

    @Test
    public void rejectTooManyTokens() throws IOException {

        AbstractModel m = TinyLlamaSuite.getOrCreate();
        String prompt = "What is the best season to plant avocados?";
        PromptContext ctx;
        {
            PromptSupport ps = m.promptSupport().get();
            ctx = ps.builder().addSystemMessage("You are a chatbot that writes short correct responses.")
                    .addUserMessage(prompt).build();
        }
        assertThrows(GenerationException.class, () -> m.generate(UUID.randomUUID(), ctx, new GeneratorParameters()
                .withSeed(42).withNtokens(5_000_000), new DoNothingGenerateEvent()));
    }

}
