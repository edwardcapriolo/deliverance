import com.codahale.metrics.MetricRegistry;
import com.fasterxml.jackson.annotation.JsonCreator;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.model.qwen2.Qwen2ModelType;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class QwenPromptTest {

    @Test
    public void qwenTest() throws IOException {
        ModelFetcher fetch = new ModelFetcher("tjake", "Qwen2.5-0.5B-Instruct-JQ4");
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache).get());
        ModelSupport.addModel("QWEN2", new Qwen2ModelType());
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),
                new MetricRegistry(), tensorCache, new KvBufferCacheSettings(true))) {
            String prompt = "What is the capital of New York, USA?";
            PromptSupport.Builder g = m.promptSupport().get().builder()
                    .addSystemMessage("You provide short answers to questions.")
                    .addUserMessage(prompt);
            Assertions.assertEquals("<|im_start|>system\n" +
                    "You provide short answers to questions.<|im_end|>\n" +
                    "<|im_start|>user\n" +
                    "What is the capital of New York, USA?<|im_end|>\n" +
                    "<|im_start|>assistant\n",g.build().getPrompt());
            var uuid = UUID.randomUUID();

            Response k = m.generate(uuid, g.build(), new GeneratorParameters().withTemperature(0.0f),(s1, f1) -> {});
            System.out.println(k);
            assertTrue(k.responseText.contains("New York City"));

        }
    }

    @Test
    public void qwenTokenize() throws IOException {
        ModelFetcher fetch = new ModelFetcher("tjake", "Qwen2.5-0.5B-Instruct-JQ4");
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache).get());
        ModelSupport.addModel("QWEN2", new Qwen2ModelType());
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),
                new MetricRegistry(), tensorCache, new KvBufferCacheSettings(true))) {
            /**
             *     >>> tokenizer("Hello world")["input_ids"]
             *     [9707, 1879]
             *
             *     >>> tokenizer(" Hello world")["input_ids"]
             *     [21927, 1879]
             */
            //List<String> x = m.getTokenizer().tokenize("Hello world");
            //Assertions.assertEquals("", x);
            //long[] k = m.getTokenizer().encode(" Hello world");
            //PromptSupport.Builder z = m.promptSupport().get().builder().addUserMessage("Hello world");
            //long[] k = m.getTokenizer().encode(z.build().getPrompt());
            //assertEquals("", Arrays.toString(k));
        }
    }

}
