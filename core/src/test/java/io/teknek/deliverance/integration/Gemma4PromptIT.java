package io.teknek.deliverance.integration;


import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.grace.AutoTokenizer;
import io.teknek.deliverance.grace.TokenIds;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.model.gemma4.Gemma4ResponseParser;
import io.teknek.deliverance.safetensors.prompt.Function;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.safetensors.prompt.Tool;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class Gemma4PromptIT {

    //@Disabled
    @Test
    public void chatWithThinking() {
        AbstractModel model = Gemma4Suite.getOrCreate();
        PromptSupport.Builder builder = model.promptSupport().get().builder()
                .addTemplateArgs(Map.of("enable_thinking", false))
                .addSystemMessage("You are a concise assistant.")
                .addUserMessage("What is the capital of New York?");
        PromptContext promptContext = builder.build();
        long[] legacyPromptTokens = model.getTokenizer().encode(promptContext.getPrompt());
        long[] runtimePromptTokens = model.encodeForRuntime(promptContext.getPrompt());
        int[] finalPromptTokens = model.constructPromptTokensForRuntime(promptContext.getPrompt());
        System.out.println("PROMPT_RENDERED_START");
        System.out.println(promptContext.getPrompt());
        System.out.println("PROMPT_RENDERED_END");
        System.out.println("LEGACY_PROMPT_TOKEN_IDS=" + Arrays.toString(legacyPromptTokens));
        System.out.println("LEGACY_PROMPT_DECODED_START");
        System.out.println(model.getTokenizer().decode(legacyPromptTokens));
        System.out.println("LEGACY_PROMPT_DECODED_END");
        System.out.println("RUNTIME_PROMPT_TOKEN_IDS=" + Arrays.toString(runtimePromptTokens));
        System.out.println("RUNTIME_PROMPT_DECODED_START");
        System.out.println(model.getPreTrainedTokenizer() == null
                ? model.getTokenizer().decode(runtimePromptTokens)
                : model.getPreTrainedTokenizer().decode(new TokenIds(Arrays.stream(runtimePromptTokens).mapToInt(v -> (int) v).toArray()), false, false, false, false));
        System.out.println("RUNTIME_PROMPT_DECODED_END");
        System.out.println("FINAL_PROMPT_TOKEN_IDS=" + Arrays.toString(finalPromptTokens));

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
                new GeneratorParameters().withTemperature(0.0f)
                        /*.withLogProbs(true).withTopLogProbs(10)*/.withMaxTokens(10),
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
        //assertTrue(parsed.content().contains("Albany"));
    }

    /*
    @Test
    public void chatWithReasoning() {
        AbstractModel model = Gemma4Suite.getOrCreate();
        PromptSupport.Builder builder = model.promptSupport().get().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                .addUserMessage("Bob is a carpenter. Sara is a teacher. Who should you call to fix your roof? Pick one of the two. Only answer one name. ");

        PromptContext promptContext = builder.build();
        Assertions.assertTrue(promptContext.toString().contains("<|think|>"));
        Response response = model.generate(
                UUID.randomUUID(),
                promptContext,
                new GeneratorParameters().withTemperature(0.0f).withMaxTokens(90),
                new GenerateEvent() {
                    @Override
                    public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                        System.out.println(next + " " + nextCleaned);
                    }
                }
        );
        System.out.println(response);
    }*/

    @Test
    public void chatWithToolTemplate() {
        AbstractModel model = Gemma4Suite.getOrCreate();
        Tool tool = Tool.from(
                Function.builder()
                        .name("get_weather")
                        .description("Gets the current weather.")
                        .addParameter("location", "string", "City and state.", true)
                        .build()
        );

        PromptContext promptContext = model.promptSupport().orElseThrow().builder()
                .addToolItem(tool)
                .addUserMessage("What is the weather in Albany, NY?")
                .build();

        assertEquals("""
                <|turn>system
                <|tool>declaration:get_weather{description:<|"|>Gets the current weather.<|"|>,parameters:{properties:{location:{description:<|"|>City and state.<|"|>,type:<|"|>STRING<|"|>}},required:[<|"|>location<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>
                <|turn>user
                What is the weather in Albany, NY?<turn|>
                <|turn>model
                """, promptContext.getPrompt());
    }
}
