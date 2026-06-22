package io.teknek.deliverance.integration;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class Gemma4PromptTemplateIT {

    @Test
    public void thinkingDisabledTemplateDoesNotRenderThinkMarker() {
        AbstractModel model = Gemma4Suite.getOrCreate();
        PromptContext promptContext = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", false))
                .addUserMessage("You have to choose between buying gold or buying silver. What do you buy?")
                .build();

        assertFalse(promptContext.getPrompt().contains("<|think|>"), promptContext.getPrompt());
        assertFalse(promptContext.getPrompt().contains("<|channel>"), promptContext.getPrompt());
        assertTrue(promptContext.getPrompt().endsWith("<|turn>model\n"), promptContext.getPrompt());
    }

    @Test
    public void thinkingEnabledTemplateRendersThinkMarker() {
        AbstractModel model = Gemma4Suite.getOrCreate();
        PromptContext promptContext = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", true))
                .addUserMessage("You have to choose between buying gold or buying silver. What do you buy?")
                .build();

        assertTrue(promptContext.getPrompt().contains("<|think|>"), promptContext.getPrompt());
        assertTrue(promptContext.getPrompt().contains("<|turn>system\n<|think|>\n<turn|>"), promptContext.getPrompt());
        assertTrue(promptContext.getPrompt().endsWith("<|turn>model\n<|channel>thought\n"), promptContext.getPrompt());
    }

    @Test
    public void runtimePromptTokenConstructionIsStable() {
        AbstractModel model = Gemma4Suite.getOrCreate();
        PromptContext promptContext = model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", false))
                .addUserMessage("You have to choose between buying gold or buying silver. What do you buy?")
                .build();

        long[] runtimeTokens = model.encodeForRuntime(promptContext.getPrompt());
        int[] finalTokens = model.constructPromptTokensForRuntime(promptContext.getPrompt());
        assertTrue(finalTokens.length == runtimeTokens.length || finalTokens.length == runtimeTokens.length + 1,
                "runtime=" + Arrays.toString(runtimeTokens) + " final=" + Arrays.toString(finalTokens));
        if (finalTokens.length == runtimeTokens.length) {
            assertArrayEquals(Arrays.stream(runtimeTokens).mapToInt(value -> (int) value).toArray(), finalTokens);
        } else {
            for (int i = 0; i < runtimeTokens.length; i++) {
                assertTrue(finalTokens[i + 1] == (int) runtimeTokens[i],
                        "runtime=" + Arrays.toString(runtimeTokens) + " final=" + Arrays.toString(finalTokens));
            }
        }
    }

    @Test
    public void rawPromptGenerationDoesNotStartWithControlChannel() {
        AbstractModel model = Gemma4Suite.getOrCreate();
        Response response = model.generate(UUID.randomUUID(), PromptContext.of("The capital of France is"),
                new GeneratorParameters().withTemperature(0.0f).withMaxTokens(8), new DoNothingGenerateEvent());

        assertFalse(response.responseTextWithSpecialTokens.startsWith("<|channel>"), response.responseTextWithSpecialTokens);
    }
}
