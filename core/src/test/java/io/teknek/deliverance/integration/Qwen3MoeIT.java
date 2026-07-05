package io.teknek.deliverance.integration;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertFalse;

@Tag("large-model")
public class Qwen3MoeIT {

    @Test
    public void qwen3MoeLoadsAndAnswersSimpleMath() {
        System.out.println("QWEN3_MOE_MATH load model start");
        try (AbstractModel model = loadBaseModel()) {
            System.out.println("QWEN3_MOE_MATH load model done");
            System.out.println("QWEN3_MOE_MATH build prompt start");
            PromptContext prompt = prompt(model, false,
                    "What is 2 + 3? Answer with only the number.");
            System.out.println("QWEN3_MOE_MATH build prompt done chars=" + prompt.getPrompt().length());

            System.out.println("QWEN3_MOE_MATH generate start");
            Response response = model.generate(UUID.randomUUID(), prompt,
                    greedy(16), printTokens("QWEN3_MOE_MATH"));
            System.out.println("QWEN3_MOE_MATH generate done");

            System.out.println("QWEN3_MOE_MATH=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
        }
    }

    @Test
    public void qwen3MoeHandlesShortCodingPrompt() {
        System.out.println("QWEN3_MOE_CODE load model start");
        try (AbstractModel model = loadBaseModel()) {
            System.out.println("QWEN3_MOE_CODE load model done");
            System.out.println("QWEN3_MOE_CODE build prompt start");
            PromptContext prompt = prompt(model, false,
                    "Write a Java method named add that returns the sum of two ints. Keep it short.");
            System.out.println("QWEN3_MOE_CODE build prompt done chars=" + prompt.getPrompt().length());

            System.out.println("QWEN3_MOE_CODE generate start");
            Response response = model.generate(UUID.randomUUID(), prompt,
                    greedy(96), printTokens("QWEN3_MOE_CODE"));
            System.out.println("QWEN3_MOE_CODE generate done");

            System.out.println("QWEN3_MOE_CODE=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
        }
    }

    @Test
    public void qwen3MoeThinkingPathProducesNonEmptyOutput() {
        System.out.println("QWEN3_MOE_THINKING load model start");
        try (AbstractModel model = loadBaseModel()) {
            System.out.println("QWEN3_MOE_THINKING load model done");
            System.out.println("QWEN3_MOE_THINKING build prompt start");
            PromptContext prompt = prompt(model, true,
                    "Think briefly, then answer in one sentence: why is prefix caching useful for LLM inference?");
            System.out.println("QWEN3_MOE_THINKING build prompt done chars=" + prompt.getPrompt().length());

            System.out.println("QWEN3_MOE_THINKING generate start");
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withTemperature(0.6f).withTopP(0.95f).withTopK(20.0f).withMaxTokens(128),
                    printTokens("QWEN3_MOE_THINKING"));
            System.out.println("QWEN3_MOE_THINKING generate done");

            System.out.println("QWEN3_MOE_THINKING=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
        }
    }

    @Test
    public void qwen3MoeQuantizeOnDemandLoadsAndGenerates() {
        ModelFetcher fetch = new ModelFetcher("Qwen", "Qwen3-30B-A3B-Base");
        System.out.println("QWEN3_MOE_QOD_MATH load model start");
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withQuantizeOnDemand(DType.Q4, "Qwen", "Qwen3-30B-A3B-Base-JQ4")
                .buildLocalTransformerModel()) {
            System.out.println("QWEN3_MOE_QOD_MATH load model done");
            System.out.println("QWEN3_MOE_QOD_MATH build prompt start");
            PromptContext prompt = prompt(model, false,
                    "What is 4 + 5? Answer with only the number.");
            System.out.println("QWEN3_MOE_QOD_MATH build prompt done chars=" + prompt.getPrompt().length());

            System.out.println("QWEN3_MOE_QOD_MATH generate start");
            Response response = model.generate(UUID.randomUUID(), prompt,
                    greedy(16), printTokens("QWEN3_MOE_QOD_MATH"));
            System.out.println("QWEN3_MOE_QOD_MATH generate done");

            System.out.println("QWEN3_MOE_QOD_MATH=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
        }
    }

    private static AbstractModel loadBaseModel() {
        ModelFetcher fetch = new ModelFetcher("Qwen", "Qwen3-30B-A3B-Base");
        return AutoModelForCausaLm.newBuilder(fetch).buildLocalTransformerModel();
    }

    private static PromptContext prompt(AbstractModel model, boolean enableThinking, String userMessage) {
        return model.promptSupport().orElseThrow().builder()
                .addTemplateArgs(Map.of("enable_thinking", enableThinking))
                .addUserMessage(userMessage)
                .build();
    }

    private static GeneratorParameters greedy(int maxTokens) {
        return new GeneratorParameters().withTemperature(0.0f).withMaxTokens(maxTokens);
    }

    private static GenerateEvent printTokens(String label) {
        return (next, nextRaw, nextCleaned, timing) -> {
            System.out.print(nextRaw == null ? "" : nextRaw);
            System.out.flush();
            System.out.printf("%n[%s token=%d elapsed=%.3fs]%n", label, next, timing);
        };
    }
}
