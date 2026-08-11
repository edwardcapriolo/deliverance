package io.teknek.deliverance.integration;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.LoraAdapter;
import io.teknek.deliverance.safetensors.fetch.LoraAdapterModelFetcher;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

/**
 * End-to-end check for Phase 2 "runtime hot-swap" LoRA support ({@link
 * AbstractModel#registerLoraAdapter}/{@link AbstractModel#setActiveAdapter}/{@link
 * AbstractModel#clearActiveAdapter}), per the step 4 plan (LoRA runtime hot-swap) Section 8.
 * Requires network access, same base/adapter pairing as {@code MergingWeightLoaderIntegrationTest}
 * (Phase 1's equivalent test) so the two are directly comparable.
 */
@Tag("small-model")
public class LoraHotSwapIT {

    private static final String PROMPT = "Hello! Who are you and what can you help me with?";

    @Test
    public void activatingAppliesTheAdapterAndClearingRestoresBaseModelBehavior() {
        ModelFetcher fetcher = new ModelFetcher("unsloth", "Llama-3.2-1B-Instruct");

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetcher).buildLocalTransformerModel()) {
            String baselineA = generate(model);

            model.registerLoraAdapter("bunnycore-chatml",
                    new LoraAdapterModelFetcher("bunnycore", "Llama-3.2-1b-chatml-lora_model"));
            model.setActiveAdapter("bunnycore-chatml");
            String adaptedResponse = generate(model);
            assertNotEquals(baselineA, adaptedResponse,
                    "activating a LoRA adapter should measurably change generation");

            model.clearActiveAdapter();
            String baselineB = generate(model);
            assertEquals(baselineA, baselineB,
                    "clearing the active adapter must restore exact base-model behavior, not just change it again");
        }
    }

    /**
     * Phase 1 (merge-at-load) vs Phase 2 (runtime hot-swap) equivalence -- only expected to hold
     * for dense working-memory types (see step 4 plan Section 1 item 6): Phase 1 computes
     * {@code quantize(base_weight + delta)} at load time, Phase 2 computes {@code
     * quantize(base_weight)} once and adds a separately-computed, full-precision delta to the
     * activation after the (unquantized, since {@code withWorkingQuantType(F32)} here) base
     * projection runs. These are different operations, not just different implementations of the
     * same one, so this comparison is meaningless under the default (I8) working-quant
     * configuration and must not be read as "Phase 1 and Phase 2 always agree."
     */
    @Test
    public void phase1MergeAtLoadAndPhase2HotSwapAgreeUnderDenseWorkingMemory() {
        ModelFetcher fetcher = new ModelFetcher("unsloth", "Llama-3.2-1B-Instruct");
        LoraAdapterModelFetcher adapterFetcher = new LoraAdapterModelFetcher("bunnycore", "Llama-3.2-1b-chatml-lora_model");

        String phase1Response;
        try (AbstractModel phase1 = AutoModelForCausaLm.newBuilder(fetcher)
                .withWorkingMemoryType(DType.F32)
                .withWorkingQuantType(DType.F32)
                .withLoraAdapter(adapterFetcher)
                .buildLocalTransformerModel()) {
            phase1Response = generate(phase1);
        }

        String phase2Response;
        try (AbstractModel phase2 = AutoModelForCausaLm.newBuilder(fetcher)
                .withWorkingMemoryType(DType.F32)
                .withWorkingQuantType(DType.F32)
                .buildLocalTransformerModel()) {
            phase2.registerLoraAdapter("bunnycore-chatml", adapterFetcher);
            phase2.setActiveAdapter("bunnycore-chatml");
            phase2Response = generate(phase2);
        }

        System.out.println("[phase1 merge-at-load] " + phase1Response);
        System.out.println("[phase2 runtime hot-swap] " + phase2Response);
        assertEquals(phase1Response, phase2Response,
                "Phase 1 and Phase 2 must agree under dense (F32) working memory -- see step 4 plan Section 1 item 6");
    }

    private static String generate(AbstractModel model) {
        PromptSupport.Builder promptBuilder = model.promptSupport().orElseThrow().builder().addUserMessage(PROMPT);
        Response response = model.generate(UUID.randomUUID(), promptBuilder.build(),
                new GeneratorParameters().withMaxTokens(80).withTemperature(0.0f).withSeed(42),
                new DoNothingGenerateEvent());
        return response.responseText;
    }
}
