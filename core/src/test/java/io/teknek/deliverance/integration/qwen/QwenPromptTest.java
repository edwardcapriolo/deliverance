package io.teknek.deliverance.integration.qwen;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.*;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.*;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.*;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class QwenPromptTest {

    @Test
    @Tag("longtest")
    public void qwenTest() throws IOException {
        ModelFetcher fetch = new ModelFetcher("tjake", "Qwen2.5-0.5B-Instruct-JQ4");
        MetricRegistry mr = new MetricRegistry();
        ArrayQueueTensorAllocator arrayQueueTensorAllocator = new ArrayQueueTensorAllocator(mr);

        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())){
            NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(arrayQueueTensorAllocator, pool).get());
        try (
             AbstractModel m = AutoModelForCausaLm.newBuilder(fetch)
                     .withWorkingMemoryType(DType.F32)
                     .withWorkingQuantType(DType.I8)
                     .withMetricRegistry(new MetricRegistry())
                     .withTensorAllocator(arrayQueueTensorAllocator)
                     .withKvBufferCacheSettings(new KvBufferCacheSettings(true))
                     .withWrappedForkJoinPool(pool)
                     .withTensorProvider(new ConfigurableTensorProvider(operation))
                     .buildLocalTransformerModel()) {
            String prompt = "What is the capital of New York, USA?";
            PromptSupport.Builder g = m.promptSupport().get().builder()
                    .addSystemMessage("You provide short answers to questions.")
                    .addUserMessage(prompt);
            assertEquals("<|im_start|>system\n" +
                    "You provide short answers to questions.<|im_end|>\n" +
                    "<|im_start|>user\n" +
                    "What is the capital of New York, USA?<|im_end|>\n" +
                    "<|im_start|>assistant\n", g.build().getPrompt());
            var uuid = UUID.randomUUID();

            Response k = m.generate(uuid, g.build(), new GeneratorParameters().withTemperature(0.0f),
                    new DoNothingGenerateEvent());
            assertTrue(k.responseText.contains("New York City"));
        }
        }
    }



}
