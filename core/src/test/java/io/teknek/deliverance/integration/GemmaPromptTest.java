package io.teknek.deliverance.integration;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.*;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class GemmaPromptTest {


    @Disabled
    public void summarizeGemmaTest() throws IOException {
        ModelFetcher fetch = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache).get());
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),
                mr, tensorCache, new KvBufferCacheSettings(true), fetch, new NoOpTokenizerRenderer())) {
            //String prompt = "Find any potential syntax errors in the code below\n";
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
            String claz = """
                      -------------------------
                      public static float cosineSimilarity(float[] a, float[] b) {
                        return "";
                      }
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
    public void chat(){
        ModelFetcher fetch = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider( new TensorCache(new MetricRegistry())).get());
        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withTensorProvider(new ConfigurableTensorProvider(operation)).build()) {
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
            PromptSupport.Builder g = model.promptSupport().get().builder()
                    .addUserMessage(prompt);
            var uuid = UUID.randomUUID();

            Response k = model.generate(uuid, g.build(), new GeneratorParameters().withTemperature(0.0f),
                    new GenerateEvent() {
                        @Override
                        public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                            System.out.println(nextCleaned);
                        }
                    });
        }
    }

    @Test
    public void gemmaTest() throws IOException {
        ModelFetcher fetch = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache).get());
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),
                mr, tensorCache, new KvBufferCacheSettings(true), fetch, new NoOpTokenizerRenderer())) {
            String prompt = "What is the capital of New York, USA?";
            PromptSupport.Builder g = m.promptSupport().get().builder()
                    .addUserMessage(prompt);
            Assertions.assertEquals("<start_of_turn>user\n" +
                    "What is the capital of New York, USA?<end_of_turn>\n" +
                    "<start_of_turn>model\n",g.build().getPrompt());
            var uuid = UUID.randomUUID();

            Response k = m.generate(uuid, g.build(), new GeneratorParameters().withTemperature(0.0f),
                    new DoNothingGenerateEvent());
            System.out.println(k.responseText);
            assertTrue(k.responseText.contains("Albany"));

            System.out.println(Arrays.toString(mr.histogram("sample.fullsample").getSnapshot().getValues()));
            System.out.println( mr.histogram("sample.fullsample").getSnapshot().getMean());
            System.out.println( mr.histogram("sample.fullsample").getSnapshot().get99thPercentile());

            System.out.println(Arrays.toString(mr.histogram("sample.forward1").getSnapshot().getValues()));
            System.out.println( mr.histogram("sample.forward1").getSnapshot().getMean());
            System.out.println( mr.histogram("sample.forward1").getSnapshot().get99thPercentile());

            System.out.println(Arrays.toString(mr.histogram("sample.dotproduct2").getSnapshot().getValues()));
            System.out.println( mr.histogram("sample.dotproduct2").getSnapshot().getMean());
            System.out.println( mr.histogram("sample.dotproduct2").getSnapshot().get99thPercentile());
        }

    }

    @Test
    public void gemmaGuidedTest() throws IOException {
        ModelFetcher fetch = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache).get());
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),
                mr, tensorCache, new KvBufferCacheSettings(true), fetch, new NoOpTokenizerRenderer())) {
            String prompt = "Who is the better NFL football team?";
            PromptSupport.Builder g = m.promptSupport().get().builder()
                    .addUserMessage(prompt);
            Assertions.assertEquals("<start_of_turn>user\n" +
                    "Who is the better NFL football team?<end_of_turn>\n" +
                    "<start_of_turn>model\n",g.build().getPrompt());
            var uuid = UUID.randomUUID();

            Response k = m.generate(uuid, g.build(), new GeneratorParameters()
                            .withTemperature(0.0f)
                            .withGuidedChoice(List.of("Giants", "Jets")),
                    new DoNothingGenerateEvent());
            System.out.println(k.responseText);
            assertTrue(k.responseText.contains("Giants"));
        }

    }

    @Test
    public void gemmaGuidedTestNeg() throws IOException {
        ModelFetcher fetch = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache).get());
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),
                mr, tensorCache, new KvBufferCacheSettings(true), fetch, new NoOpTokenizerRenderer())) {
            String prompt = "Which NFL franchise does not play in New York?";
            PromptSupport.Builder g = m.promptSupport().get().builder()
                    .addUserMessage(prompt);
            var uuid = UUID.randomUUID();

            Response k = m.generate(uuid, g.build(), new GeneratorParameters()
                            .withTemperature(0.0f)
                            .withGuidedChoice(List.of("Giants", "Jets", "Seahawks")),
                    new GenerateEvent() {
                        @Override
                        public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                            System.out.println(nextCleaned);
                        }
                    });
            System.out.println(k.responseText);
            assertTrue(k.responseText.contains("Seahawks"));
        }

    }





}
