package io.teknek.deliverance.integration;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

@Tag("large-model")
public class Qwen3SmallIT {

    @Test
    public void qwen306BLoadsAndGeneratesShortAnswer() {
        ModelFetcher fetch = new ModelFetcher("Qwen", "Qwen3-0.6B");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).buildLocalTransformerModel()) {
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addTemplateArgs(Map.of("enable_thinking", false))
                    .addUserMessage("What is 1 + 1? Answer with only the number.")
                    .build();
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withTemperature(0.0f).withMaxTokens(8), new DoNothingGenerateEvent());
            System.out.println("QWEN3_06B_SHORT=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
        }
    }

    @Test
    public void qwen306BThinkingPathProducesNonEmptyOutput() {
        ModelFetcher fetch = new ModelFetcher("Qwen", "Qwen3-0.6B");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).buildLocalTransformerModel()) {
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addTemplateArgs(Map.of("enable_thinking", true))
                    .addUserMessage("What is 1 + 1? Think briefly, then answer.")
                    .build();
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withTemperature(0.0f).withMaxTokens(256), new DoNothingGenerateEvent());
            System.out.println("QWEN3_06B_THINKING=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
            assertEquals("""
                    <think>
                    Okay, so the question is 1 plus 1. Hmm, let's think. When you add two numbers together, you're combining them. So 1 plus 1 would be like taking one and adding another one. Let me visualize this. If I have a number line, starting at 1 and then adding another 1, I would end up at 2. That makes sense. But wait, is there any trick here? Like, maybe some people think of it in a different way? For example, sometimes people might confuse addition with multiplication or something else. But no, 1 plus 1 is straightforward. It's just two ones added together. So the answer should be 2. I don't think there's any hidden meaning here. It's a simple arithmetic problem.
                    </think>
                    
                    1 + 1 equals 2.
                    """.trim(), response.responseText);
        }
    }
}
