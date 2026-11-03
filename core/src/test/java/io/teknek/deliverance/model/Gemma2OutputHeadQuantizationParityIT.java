package io.teknek.deliverance.model;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class Gemma2OutputHeadQuantizationParityIT {

    @Test
    public void q4OutputHeadPreservesFirstTokenAndTopLogitsForFixedPrompts() {
        ModelFetcher fetcher = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
        try (AbstractModel denseOutput = AutoModelForCausaLm.newBuilder(fetcher).buildLocalTransformerModel();
             AbstractModel q4Output = AutoModelForCausaLm.newBuilder(fetcher)
                     .withOutputHeadQuantization(DType.Q4)
                     .buildLocalTransformerModel()) {
            for (String prompt : List.of(
                    "What is 1 + 1?",
                    "Explain tensor parallelism in one sentence.",
                    "Write a Python function that returns the larger of two integers.")) {
                int denseGenerated = denseOutput.generate(UUID.randomUUID(), PromptContext.of(prompt),
                        new GeneratorParameters().withTemperature(0.0f).withMaxTokens(1).withSeed(123),
                        new DoNothingGenerateEvent()).generatedTokens.getFirst();
                int q4Generated = q4Output.generate(UUID.randomUUID(), PromptContext.of(prompt),
                        new GeneratorParameters().withTemperature(0.0f).withMaxTokens(1).withSeed(123),
                        new DoNothingGenerateEvent()).generatedTokens.getFirst();
                assertEquals(denseGenerated, q4Generated, "generated first token changed for prompt: " + prompt);
            }
        }
    }

}
