package io.teknek.deliverance.integration;

import io.teknek.deliverance.generator.FinishReason;
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

import java.util.Map;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

@Tag("large-model")
class Qwen3GuidedGrammarIT {
    @Test
    void qwen306BGeneratesSimpleToonListWithGuidedGrammar() {
        ModelFetcher fetch = new ModelFetcher("Qwen", "Qwen3-0.6B");
        Assumptions.assumeTrue(fetch.pathForModel().toFile().isDirectory(),
                "Qwen/Qwen3-0.6B cache is not present; skipping guided grammar smoke test");
        String expected = "users[2]{id,name}:\n  1,Ada\n  2,Bob";
        String grammar = """
                root ::= users
                users ::= "users[2]{id,name}:\\n" row1 "\\n" row2
                row1 ::= "  1,Ada"
                row2 ::= "  2,Bob"
                """;

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).buildLocalTransformerModel()) {
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addTemplateArgs(Map.of("enable_thinking", false))
                    .addUserMessage("Return a TOON users table with two rows. Use id and name columns only.")
                    .build();
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters()
                            .withTemperature(0.0f)
                            .withMaxTokens(64)
                            .withGuidedGrammar(grammar),
                    new DoNothingGenerateEvent());

            System.out.println("QWEN3_GUIDED_TOON=" + response.responseTextWithSpecialTokens.replace("\n", "\\n"));
            assertEquals(FinishReason.STOP_TOKEN, response.finishReason);
            assertEquals(expected, response.responseText);
        }
    }
}
