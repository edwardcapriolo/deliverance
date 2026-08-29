package io.teknek.deliverance.integration.qwen;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("large-model")
class NemotronBenchmarkPromptHeadQuantizationTest {
    private static final String OWNER = "nvidia";
    private static final String MODEL = "Nemotron-Labs-Diffusion-3B-Base-JQ4";
    private static final String MATH_PROMPT = "Solve step by step. A bus starts with an unknown number of passengers. "
            + "At the first stop, half get off and 4 get on. At the second stop, 6 get off and 8 get on. "
            + "There are 25 passengers heading to the third stop. How many passengers started at the terminal? "
            + "Then compute total fare collected if every person who ever boarded paid $2.";

    @Test
    void arBenchmarkMathPromptWithDenseHeadDoesNotImmediatelyStop() {
        ModelFetcher fetch = new ModelFetcher(OWNER, MODEL).withDownload(false);
        Assumptions.assumeTrue(fetch.pathForModel().toFile().isDirectory(),
                MODEL + " cache is not present: " + fetch.pathForModel());

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withDownload(false)
                .withGenerationOptions(Map.of("mode", "ar"))
                .buildLocalTransformerModel()) {
            PromptContext prompt = benchmarkPrompt(model.promptSupport(), MATH_PROMPT);
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withTemperature(0.0f).withMaxTokens(16).withSeed(42),
                    new DoNothingGenerateEvent());

            System.out.printf(java.util.Locale.ROOT,
                    "[nemotron-head-diagnostic] head=dense prompt_tokens=%d generated=%d finish=%s text=%s%n",
                    response.promptTokens, response.generatedTokens.size(), response.finishReason,
                    response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            assertTrue(response.generatedTokens.size() > 2,
                    "dense output head also stopped immediately: " + response.responseTextWithSpecialTokens);
        }
    }

    private static PromptContext benchmarkPrompt(Optional<PromptSupport> promptSupport, String prompt) {
        if (promptSupport.isPresent()) {
            return promptSupport.get().builder().addUserMessage(prompt).build();
        }
        return PromptContext.of("user: " + prompt + "\nassistant: ");
    }
}
