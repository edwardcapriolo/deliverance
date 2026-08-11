package io.teknek.deliverance.model.qwen;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.Function;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.safetensors.prompt.Tool;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Disabled;

import java.io.IOException;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class QwenTest {

    @Disabled("Only single file tensors are supported")
    public void sample() throws IOException {
        ModelFetcher fetch = new ModelFetcher("Qwen", "Qwen2.5-7B");
        com.codahale.metrics.MetricRegistry mr = new com.codahale.metrics.MetricRegistry();
        ArrayQueueTensorAllocator arrayQueueTensorAllocator = new ArrayQueueTensorAllocator(mr);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            try (AbstractModel m = AutoModelForCausaLm.newBuilder(fetch)
                    .withWorkingMemoryType(DType.F32)
                    .withWorkingQuantType(DType.I8)
                    .withTensorAllocator(arrayQueueTensorAllocator)
                    .withWrappedForkJoinPool(pool)
                    .withTensorProvider(new ConfigurableTensorProvider(arrayQueueTensorAllocator, pool))
                    .buildLocalTransformerModel()) {
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
                }
                {
                    PromptSupport ps = m.promptSupport().get();
                    Tool t = Tool.from(Function.builder().name("hello").build());
                    ctx = ps.builder().addSystemMessage("You are a chatbot that writes short correct responses.")
                            .addUserMessage(prompt).build(List.of(t));
                    String expected = """
                            <|system|>
                            You are a chatbot that writes short correct responses.</s>
                            <|user|>
                            What is the best season to plant avocados?</s>
                            <|assistant|>
                            """;
                    assertEquals(expected, ctx.getPrompt());// it does not change the prompt to have tools

                    Response r = m.generate(UUID.randomUUID(), ctx, new GeneratorParameters().withSeed(42)
                            , new DoNothingGenerateEvent());
                    System.out.println(r);
                    assertTrue(mr.meter("tensorcache.dirtyget").getCount() > 100);
                    mr.getMeters().entrySet().stream().forEach(x -> System.out.println(x.getKey() + " " + x.getValue().getCount()));

                /*
                assertEquals("""
                        The best thing to do is to look for the plant that best suits your needs and preferences. Avocados are a popular fruit that are grown in many regions around the world. Some of the best regions for avocado production include California, Mexico, and Peru.
                        """, r.responseText);*/
                }
            }
        }
    }
}
