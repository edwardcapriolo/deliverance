package io.teknek.deliverance.integration;


import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.TokenIds;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.model.gemma4.Gemma4ResponseParser;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertTrue;


public class Gemma4PromptIT {

    @Disabled
    //@Test
    public void chatWithThinking() {
        AbstractModel model = Gemma4Suite.getOrCreate();
        PromptSupport.Builder builder = model.promptSupport().get().builder()
                //.addTemplateArgs(Map.of("enable_thinking", true))
                .addSystemMessage("You are a concise assistant.")
                .addUserMessage("What is the capital of New York?");
        PromptContext promptContext = builder.build();
        long[] promptTokens = model.getTokenizer().encode(promptContext.getPrompt());
        System.out.println("PROMPT_RENDERED_START");
        System.out.println(promptContext.getPrompt());
        System.out.println("PROMPT_RENDERED_END");
        System.out.println("PROMPT_TOKEN_IDS=" + Arrays.toString(promptTokens));
        System.out.println("PROMPT_DECODED_START");
        System.out.println(model.getTokenizer().decode(promptTokens));
        System.out.println("PROMPT_DECODED_END");

        var graceTokenizer = AutoTokenizer.fromPretrained(
                new AutoTokenizer.OwnerNameOrPath(new AutoTokenizer.OwnerName("google", "gemma-4-E2B-it")));

        var graceEncoding = graceTokenizer.encode(promptContext.getPrompt());
        TokenIds graceTokenIds = new TokenIds(graceEncoding.inputIds());
        System.out.println("GRACE_PROMPT_TOKEN_IDS=" + Arrays.toString(Arrays.stream(graceEncoding.inputIds()).asLongStream().toArray()));
        System.out.println("GRACE_PROMPT_DECODED_START");
        System.out.println(graceTokenizer.decode(graceTokenIds, false, false, false, false));
        System.out.println("GRACE_PROMPT_DECODED_END");

        Response response = model.generate(
                UUID.randomUUID(),
                promptContext,
                new GeneratorParameters().withTemperature(0.0f).withLogProbs(true).withTopLogProbs(10).withMaxTokens(10),
                new GenerateEvent() {
                    @Override
                    public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                        System.out.println(next + " " + nextCleaned);
                    }
                }
        );
        Gemma4ResponseParser.Parsed parsed = Gemma4ResponseParser.parse(
                response.responseTextWithSpecialTokens,
                response.responseText
        );
        System.out.println(response);
        assertTrue(parsed.content().contains("Albany"));
    }
}
