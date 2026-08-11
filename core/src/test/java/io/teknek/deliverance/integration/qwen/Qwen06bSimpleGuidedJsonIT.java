package io.teknek.deliverance.integration.qwen;

import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DefaultCausalLanguageModel;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.sketches.SketchesSettings;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Qwen06bSimpleGuidedJsonIT {

    private static final String FOOD_SCHEMA = """
            {"type":"object","additionalProperties":false,"required":["a"],"properties":{"a":{"type":"string","pattern":"[A-Za-z][A-Za-z ]{0,39}"}}}
            """;
    private static final String GENERIC_FOOD_SCHEMA = """
            {"type":"object","additionalProperties":false,"required":["a"],"properties":{"a":{"type":"string","maxLength":40}}}
            """;

    @Test
    void simpleFoodGuidedJsonProducesSaneStringValues() throws Exception {
        ModelFetcher fetch = new ModelFetcher("edwardcapriolo", "Qwen3-0.6B-JQ4").withDownload(false);
        Assumptions.assumeTrue(fetch.pathForModel().toFile().isDirectory(),
                "Qwen3-0.6B-JQ4 cache is not present: " + fetch.pathForModel());

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).withDownload(false).buildLocalTransformerModel()) {
            int[] seeds = {1, 2, 3, 4, 5, 42, 123, 777, 2026, 719081789};
            for (int seed : seeds) {
                PromptContext prompt = model.promptSupport().orElseThrow().builder()
                        .addTemplateArgs(Map.of("enable_thinking", false))
                        .addSystemMessage("Return only compact JSON matching the schema.")
                        .addUserMessage("Tell me one food you like.")
                        .build();
                Response response = model.generate(UUID.randomUUID(), prompt,
                        new GeneratorParameters()
                                .withSeed(seed)
                                .withTemperature(0.8f)
                                .withMaxTokens(80)
                                .withGuidedJson(FOOD_SCHEMA),
                        new DoNothingGenerateEvent());

                String value = JsonUtils.om.readTree(response.responseText).path("a").asText();
                System.out.printf("QWEN_06B_SIMPLE_GUIDED_JSON seed=%d finish=%s tokens=%d text=%s value=%s%n",
                        seed, response.finishReason, response.generatedTokens.size(),
                        response.responseText.replace("\n", "\\n"), value.replace("\n", "\\n"));

                assertFalse(value.isBlank(), "blank value for seed " + seed + ": " + response.responseText);
                assertFalse(value.matches("^[,;:._\\-\\s}\\]\\[{]+.*"),
                        "leading delimiter value for seed " + seed + ": " + response.responseText);
                assertFalse(value.matches(".*[\\p{Cntrl}&&[^\\r\\n\\t]].*"),
                        "control character value for seed " + seed + ": " + response.responseText);
                assertTrue(value.matches(".*[A-Za-z].*"),
                        "value has no latin letters for seed " + seed + ": " + response.responseText);
            }
        }
    }

    @Test
    void eagerAndLazyGuidesReturnSameGenericFoodString() throws Exception {
        ModelFetcher fetch = new ModelFetcher("edwardcapriolo", "Qwen3-0.6B-JQ4").withDownload(false);
        Assumptions.assumeTrue(fetch.pathForModel().toFile().isDirectory(),
                "Qwen3-0.6B-JQ4 cache is not present: " + fetch.pathForModel());

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).withDownload(false).buildLocalTransformerModel()) {
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addTemplateArgs(Map.of("enable_thinking", false))
                    .addSystemMessage("Return only compact JSON matching the schema.")
                    .addUserMessage("Tell me one food you like.")
                    .build();

            Response eager = DefaultCausalLanguageModel.local(model, sketches(SketchesSettings.GuidedIndexMode.EAGER))
                    .generate(UUID.randomUUID(), prompt, genericParams(), new DoNothingGenerateEvent());
            Response lazy = DefaultCausalLanguageModel.local(model, sketches(SketchesSettings.GuidedIndexMode.LAZY))
                    .generate(UUID.randomUUID(), prompt, genericParams(), new DoNothingGenerateEvent());

            System.out.println("QWEN_06B_GENERIC_GUIDED_JSON_EAGER=" + eager.responseText.replace("\n", "\\n"));
            System.out.println("QWEN_06B_GENERIC_GUIDED_JSON_LAZY=" + lazy.responseText.replace("\n", "\\n"));
            assertEquals(eager.responseText, lazy.responseText);
        }
    }

    private static GeneratorParameters genericParams() {
        return new GeneratorParameters()
                .withSeed(1)
                .withTemperature(0.8f)
                .withMaxTokens(80)
                .withGuidedJson(GENERIC_FOOD_SCHEMA);
    }

    private static SketchesSettings sketches(SketchesSettings.GuidedIndexMode mode) {
        return new SketchesSettings(10_000, 10_000, 20_000_000, mode);
    }

}
