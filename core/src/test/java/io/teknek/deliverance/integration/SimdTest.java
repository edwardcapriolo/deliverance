package io.teknek.deliverance.integration;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.*;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class SimdTest {

    AbstractTensor allOnes(int size) {
        AbstractTensor f = new FloatBufferTensor(1, size);
        for (int i = 0; i < size; i++) {
            f.set(1.0f, 0, i);
        }
        return f;
    }

    @Test
    void goTryIt(){
        ArrayQueueTensorAllocator tc = new ArrayQueueTensorAllocator(new MetricRegistry());
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            NativeSimdTensorOperations n = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tc, pool).get());
            int size = 1024;
            NaiveTensorOperations controlOps = new NaiveTensorOperations();
            AbstractTensor a = allOnes(size);
            AbstractTensor b = allOnes(size);
            float control = controlOps.dotProduct(a, b, size);
            assertEquals(control, 1024f);
            assertEquals(control, n.dotProduct(a, b, size));
        }
    }

    @Disabled
    public void sample() throws IOException {
        String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        try (AbstractModel m = AutoModelForCausaLm.newBuilder(fetch).buildLocalTransformerModel()) {
            String prompt = "What is the best season to plant avocados?";
            PromptContext ctx;
            {
                PromptSupport ps = m.promptSupport().get();
                ctx = ps.builder().addSystemMessage("You are a chatbot that writes short correct responses.")
                        .addUserMessage(prompt).build();
                String expected = """
                        <|system|>
                        You are a chatbot that writes short correct responses.</s>
                        <|user|>
                        What is the best season to plant avocados?</s>
                        <|assistant|>
                        """;
                assertEquals(expected, ctx.getPrompt());
                Response r = m.generate(UUID.randomUUID(), ctx, new GeneratorParameters().withSeed(43),
                        new DoNothingGenerateEvent());
                System.out.println(r.responseText);
            }
        }
    }
}
