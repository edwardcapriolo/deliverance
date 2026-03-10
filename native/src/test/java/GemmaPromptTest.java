import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class GemmaPromptTest {

    @Test
    public void gemmaTest() throws IOException {
        ModelFetcher fetch = new ModelFetcher("tjake", "gemma-2-2b-it-JQ4");
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache).get());
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),
                mr, tensorCache, new KvBufferCacheSettings(true), fetch)) {
            String prompt = "What is the capital of New York, USA?";
            PromptSupport.Builder g = m.promptSupport().get().builder()
                    //.addSystemMessage("You provide short answers to questions.")
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
                mr, tensorCache, new KvBufferCacheSettings(true), fetch)) {
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
                mr, tensorCache, new KvBufferCacheSettings(true), fetch)) {
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
