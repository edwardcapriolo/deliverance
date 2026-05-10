package io.teknek.deliverance.model;

import com.codahale.metrics.ConsoleReporter;
import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class MaxTokenTest {

    @Test
    public void maxTokens() {
        ModelFetcher fetch = new ModelFetcher("tjake", "Llama-3.2-1B-Instruct-JQ4");
        var uuid = UUID.randomUUID();
        MetricRegistry registry = new MetricRegistry();
        /*
        So yes: I do know why. This test is unusually sensitive because it is an open-ended creative prompt on a small
        quantized model with greedy decode, where tiny numeric differences can change the whole story.
         */

        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
             AbstractModel m = AutoModelForCausaLm.newBuilder(fetch)
                .withMetricRegistry(registry)
                .withTensorProvider(new ConfigurableTensorProvider(new ArrayQueueTensorAllocator(registry), pool)).build()) {
            String prompt = "Construct a short story about a Java developer who takes on all of python and rust community";
            PromptContext ctx = m.promptSupport().get().builder()
                    .addUserMessage(prompt)
                    .build();
                Response k = m.generate(uuid, ctx, new GeneratorParameters()
                    .withNtokens(2048)
                    .withMaxTokens(17)
                    .withIncludeStopStrInOutput(false)
                    .withStopWords(List.of("<|eot_id|>"))
                    .withTemperature(0.0f).withSeed(99998), new DoNothingGenerateEvent());
            assertEquals(17, k.generatedTokens.size());
            /*
            assertEquals("**The Java Developer's Quest**\n" +
                    "\n" +
                    "In a world where technology was the ultimate force,", k.responseText);
             */
            assertTrue(k.responseText.contains("Java"));

        }
    }
}
