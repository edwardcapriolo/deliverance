package io.teknek.deliverance.model;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MaxTokenTest {

    @Test
    public void maxTokens() {
        ModelFetcher fetch = new ModelFetcher("tjake", "Llama-3.2-1B-Instruct-JQ4");
        var uuid = UUID.randomUUID();

        try (AbstractModel m = AutoModelForCausaLm.newBuilder(fetch).withWorkingQuantType(DType.I8)
                .withTokenTokenRenderer(new TokenizerRenderer())
                .withTensorProvider(new ConfigurableTensorProvider(new TensorCache(new MetricRegistry()))).build()) {
            String prompt = "Construct a short story about a Java developer who takes on all of python and rust community";
            PromptContext ctx = m.promptSupport().get().builder()
                    .addUserMessage(prompt)
                    .build();
            Response k = m.generate(uuid, ctx, new GeneratorParameters()
                    .withNtokens(2048)
                    .withMaxTokens(17)
                    .withIncludeStopStrInOutput(false)
                    .withStopWords(List.of("<|eot_id|>"))
                    .withTemperature(0.2f).withSeed(99998), new DoNothingGenerateEvent());
            assertEquals(17, k.generatedTokens.size());
            assertEquals("Here's a short story about a Java developer who takes a significant role in the Rust", k.responseText);
        }
    }
}
