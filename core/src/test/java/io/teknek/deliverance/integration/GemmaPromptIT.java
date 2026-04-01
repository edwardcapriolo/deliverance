package io.teknek.deliverance.integration;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.*;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GemmaPromptIT {


    @Disabled
    public void summarizeGemmaTest() throws IOException {
        ModelFetcher fetch = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache, pool).get());
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),
                mr, tensorCache, new KvBufferCacheSettings(true), fetch, new NoOpTokenizerRenderer(), new DefaultToolCallParser(), pool)) {
            String prompt = """
                    You are a software engineer.
                    
                    ### INSTRUCTIONS ###
                    *   Your task is to write a complete, correct, and production-ready Java code.
                    *   Do not include any explanations, comments, or surrounding text, only the code block.
                    *   You MUST use the "java" markdown code block format.
                    
                    ### CODE ###
                    Implement the method:
                    public static float cosineSimilarity(float[] a, float[] b)
                    """;
            PromptSupport.Builder g = m.promptSupport().get().builder()
                    .addUserMessage(prompt);

            var uuid = UUID.randomUUID();

            Response k = m.generate(uuid, g.build(), new GeneratorParameters().withNtokens(8192).withTemperature(0.0f),
                    new GenerateEvent() {
                        @Override
                        public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                            System.out.println(nextCleaned);
                        }
                    });
            System.out.println(k.responseText);

        }

    }

    //    return np.full(
    //        (1, (vocab_size + 31) // 32),
    //        -1,
    //        dtype=np.int32,
    //    )
    @Test
    public void chat() {

        AbstractModel model = Gemma2Suite.getOrCreate();
        String prompt = """
                What does this python code do?
                ---------------------------
                def allocate_token_bitmask(vocab_size: int) -> np.ndarray:
                    return np.full(
                        (1, (vocab_size + 31) // 32),
                        -1,
                        dtype=np.int32,
                    )
                """;
        PromptSupport.Builder g = model.promptSupport().get().builder().addUserMessage(prompt);
        var uuid = UUID.randomUUID();

        Response k = model.generate(uuid, g.build(), new GeneratorParameters().withMaxTokens(100).withTemperature(0.0f),
               new DoNothingGenerateEvent());
        String expected = """
Let's break down this Python code snippet.

**Understanding the Code**

This code defines a function called `allocate_token_bitmask` that generates a bitmask for a vocabulary.  Here's a step-by-step explanation:

1. **Function Definition:**
   - `def allocate_token_bitmask(vocab_size: int) -> np.ndarray:`
     - This line defines a function named `allocate_token_bitmask`. """.trim();
        assertEquals(expected, k.responseText.trim());

    }

    @Test
    public void gemmaTest() throws IOException {
        AbstractModel m = Gemma2Suite.getOrCreate();
        MetricRegistry mr = Gemma2Suite.getBuilder().getMr();
        String prompt = "What is the capital of New York, USA?";
        PromptSupport.Builder g = m.promptSupport().get().builder()
                .addUserMessage(prompt);
        assertEquals("<start_of_turn>user\n" +
                "What is the capital of New York, USA?<end_of_turn>\n" +
                "<start_of_turn>model\n", g.build().getPrompt());
        var uuid = UUID.randomUUID();

        Response k = m.generate(uuid, g.build(), new GeneratorParameters().withTemperature(0.0f),
                new DoNothingGenerateEvent());
        System.out.println(k.responseText);
        assertTrue(k.responseText.contains("Albany"));

        System.out.println(Arrays.toString(mr.histogram("sample.fullsample").getSnapshot().getValues()));
        System.out.println(mr.histogram("sample.fullsample").getSnapshot().getMean());
        System.out.println(mr.histogram("sample.fullsample").getSnapshot().get99thPercentile());

        System.out.println(Arrays.toString(mr.histogram("sample.forward1").getSnapshot().getValues()));
        System.out.println(mr.histogram("sample.forward1").getSnapshot().getMean());
        System.out.println(mr.histogram("sample.forward1").getSnapshot().get99thPercentile());

        System.out.println(Arrays.toString(mr.histogram("sample.dotproduct2").getSnapshot().getValues()));
        System.out.println(mr.histogram("sample.dotproduct2").getSnapshot().getMean());
        System.out.println(mr.histogram("sample.dotproduct2").getSnapshot().get99thPercentile());

    }

    @Test
    public void gemmaGuidedTest() {
        AbstractModel m = Gemma2Suite.getOrCreate();
        String prompt = "Who is the better NFL football team?";
        PromptSupport.Builder g = m.promptSupport().get().builder()
                .addUserMessage(prompt);
        assertEquals("<start_of_turn>user\n" +
                "Who is the better NFL football team?<end_of_turn>\n" +
                "<start_of_turn>model\n", g.build().getPrompt());
        var uuid = UUID.randomUUID();
        Response k = m.generate(uuid, g.build(), new GeneratorParameters()
                        .withTemperature(0.0f)
                        .withGuidedChoice(List.of("Giants", "Jets")),
                new DoNothingGenerateEvent());
        System.out.println(k.responseText);
        assertTrue(k.responseText.contains("Giants"));
    }

    @Test
    public void gemmaGuidedTestNeg() throws IOException {
        AbstractModel m = Gemma2Suite.getOrCreate();
        String prompt = "Which NFL franchise does not play in New York?";
        PromptSupport.Builder g = m.promptSupport().get().builder()
                .addUserMessage(prompt);
        var uuid = UUID.randomUUID();

        Response k = m.generate(uuid, g.build(), new GeneratorParameters()
                        .withTemperature(0.0f)
                        .withGuidedChoice(List.of("Giants", "Jets", "Seahawks")),
               new DoNothingGenerateEvent());
        assertTrue(k.responseText.contains("Seahawks"));
    }

}
