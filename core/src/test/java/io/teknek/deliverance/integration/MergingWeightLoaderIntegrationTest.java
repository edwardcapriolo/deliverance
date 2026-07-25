package io.teknek.deliverance.integration;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.fetch.LoraAdapterModelFetcher;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Test;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertNotEquals;

/**
 * End-to-end check for Phase 1 "merge-at-load" LoRA support ({@link
 * io.teknek.deliverance.safetensors.MergingWeightLoader} wired through {@link
 * AutoModelForCausaLm.Builder#withLoraAdapter}), per {@code
 * StepPlans/deliverance_lora_step3_merging_weightloader_plan_v1.md} Section 7. Requires network
 * access to download the base model and adapter, same as the other tests in this package.
 *
 * <p>Uses {@code unsloth/Llama-3.2-1B-Instruct} (dense HF format, required by {@link
 * io.teknek.deliverance.safetensors.MergingWeightLoader}'s dtype guard) paired with {@code
 * bunnycore/Llama-3.2-1b-chatml-lora_model}, the same base/adapter pairing already verified in
 * {@code LoraAdapterTest} -- deliberately not {@code tjake/Llama-3.2-1B-Instruct-JQ4}, which is
 * pre-quantized on disk and out of scope for merge-at-load (Section 1 item 4 of the step 3
 * plan).</p>
 */
public class MergingWeightLoaderIntegrationTest {

    private static final String PROMPT = "Hello! Who are you and what can you help me with?";

    @Test
    public void loraAdaptedGenerationDiffersFromBaseModel() {
        ModelFetcher fetcher = new ModelFetcher("unsloth", "Llama-3.2-1B-Instruct");

        String baseResponse;
        try (AbstractModel base = AutoModelForCausaLm.newBuilder(fetcher).buildLocalTransformerModel()) {
            baseResponse = generate(base);
        }

        String adaptedResponse;
        try (AbstractModel adapted = AutoModelForCausaLm.newBuilder(fetcher)
                .withLoraAdapter(new LoraAdapterModelFetcher("bunnycore", "Llama-3.2-1b-chatml-lora_model"))
                .buildLocalTransformerModel()) {
            adaptedResponse = generate(adapted);
        }

        System.out.println("[base] " + baseResponse);
        System.out.println("[lora-adapted] " + adaptedResponse);
        assertNotEquals(baseResponse, adaptedResponse,
                "LoRA-adapted generation should measurably differ from the un-adapted base model");
    }

    private static String generate(AbstractModel model) {
        PromptSupport.Builder promptBuilder = model.promptSupport().orElseThrow().builder().addUserMessage(PROMPT);
        Response response = model.generate(UUID.randomUUID(), promptBuilder.build(),
                new GeneratorParameters().withMaxTokens(80).withTemperature(0.0f).withSeed(42),
                new DoNothingGenerateEvent());
        return response.responseText;
    }
}
