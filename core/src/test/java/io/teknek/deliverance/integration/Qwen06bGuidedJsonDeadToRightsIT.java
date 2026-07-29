package io.teknek.deliverance.integration;

import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

@Tag("large-model")
class Qwen06bGuidedJsonDeadToRightsIT {

    @Test
    void qwen06bGuidedJsonCaseSetupProducesParseableJson() {
        ModelFetcher fetch = new ModelFetcher("edwardcapriolo", "Qwen3-0.6B-JQ4").withDownload(false);
        Assumptions.assumeTrue(fetch.pathForModel().toFile().isDirectory(),
                "Qwen3-0.6B-JQ4 cache is not present: " + fetch.pathForModel());

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).withDownload(false).buildLocalTransformerModel()) {
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addTemplateArgs(Map.of("enable_thinking", false))
                    .addSystemMessage(systemPrompt())
                    .addUserMessage(caseSetupPrompt("Lysandra", "garden center", "camera"))
                    .build();

            int[] seeds = {690171677, 42, 123, 777, 2026, 8675309, 31415926, 27182818};
            for (int seed : seeds) {
                Response response = model.generate(UUID.randomUUID(), prompt,
                        params(seed), new DoNothingGenerateEvent());

                System.out.println("QWEN_06B_DTR_GUIDED_JSON_SEED=" + seed);
                System.out.println("QWEN_06B_DTR_GUIDED_JSON_PROMPT=" + prompt.getPrompt().replace("\n", "\\n"));
                System.out.println("QWEN_06B_DTR_GUIDED_JSON_OUTPUT="
                        + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
                System.out.println("QWEN_06B_DTR_GUIDED_JSON_TEXT="
                        + response.responseText.replace("\n", "\\n"));
                System.out.println("QWEN_06B_DTR_GUIDED_JSON_TOKENS=" + response.generatedTokens.size());

                try {
                    JsonUtils.om.readTree(response.responseText);
                } catch (Exception e) {
                    fail("guided_json produced invalid JSON for seed " + seed + ":\n" + response.responseText, e);
                }
            }
        }
    }

    private static GeneratorParameters params(int seed) {
        return new GeneratorParameters()
                .withSeed(seed)
                .withTemperature(0.8f)
                .withXtcThreshold(0.1f)
                .withXtcProbability(0.2f)
                .withMaxTokens(384)
                .withGuidedJson(caseFileSchemaJson());
    }

    private static String caseSetupPrompt(String suspectName, String place, String item) {
        return "Start a new case for Dead to Rights.\n"
                + "The suspect is: " + suspectName + "\n"
                + "The setting is: " + place + "\n"
                + "The suspect stole this item: " + item + "\n"
                + "The public fields are caseTitle, suspect, setting, meansClue, opportunityClue, and mistakeClue.\n"
                + "Do not include a public crime summary field. The public case should be ambiguous and clue-driven.\n"
                + "The hiddenTruth object contains the actual crime, method, mistakes, and why the clues matter.\n"
                + "A clue must be a concrete observable fact, such as a receipt, timestamp, key, witness statement, misplaced object, log entry, footprint, note, damaged lock, altered record, or contradiction.\n"
                + "Do not use the suspect name, setting, or stolen item alone as a clue.\n"
                + "meansClue must show how the suspect could access or take the item.\n"
                + "opportunityClue must place the suspect near the item at the relevant time.\n"
                + "mistakeClue must show an error the suspect made while hiding or covering up the theft.\n"
                + "The crime is theft. The theft went wrong in three ways, creating the three clues.\n"
                + "The suspect either left evidence, was seen by a witness, contradicted a timeline, had unusual access, or hid the item poorly.";
    }

    private static String systemPrompt() {
        return "Dead to Rights is a light, non-violent mystery game. "
                + "The suspect is guilty of a small theft and should hide the truth until cornered.";
    }

    private static String caseFileSchemaJson() {
        Map<String, Object> string = Map.of("type", "string");
        Map<String, Object> schema = Map.of(
                "type", "object",
                "additionalProperties", false,
                "required", List.of("caseTitle", "suspect", "setting", "meansClue",
                        "opportunityClue", "mistakeClue", "hiddenTruth"),
                "properties", Map.of(
                        "caseTitle", string,
                        "suspect", string,
                        "setting", string,
                        "meansClue", string,
                        "opportunityClue", string,
                        "mistakeClue", string,
                        "hiddenTruth", Map.of(
                                "type", "object",
                                "additionalProperties", false,
                                "required", List.of("crime", "method", "mistakes", "whyCluesMatter"),
                                "properties", Map.of(
                                        "crime", string,
                                        "method", string,
                                        "mistakes", Map.of("type", "array", "items", string),
                                        "whyCluesMatter", Map.of("type", "array", "items", string)))));
        try {
            return JsonUtils.om.writeValueAsString(schema);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
