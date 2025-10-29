import com.codahale.metrics.MetricRegistry;
import com.github.tjake.jlama.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.safetensors.prompt.Function;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.safetensors.prompt.Tool;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class SimdTest {

    static AbstractTensor allOnes(int size) {
        AbstractTensor f = new FloatBufferTensor(1, size);
        for (int i = 0; i < size; i++) {
            f.set(1.0f, 0, i);
        }
        return f;
    }

    @Test
    void goTryIt(){
        System.load("/home/edward/deliverence/native/target/native-lib-only/libdeliverance.so");
        TensorCache tc = new TensorCache(new MetricRegistry());
        NativeSimdTensorOperations n = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tc).get());
        int size = 1024;
        NaiveTensorOperations controlOps = new NaiveTensorOperations();
        AbstractTensor a = allOnes(size);
        AbstractTensor b = allOnes(size);
        float control = controlOps.dotProduct(a, b, size);
        assertEquals(control, 1024f);
        assertEquals(control, n.dotProduct(a,b,size));
    }

    @Test
    public void sample() throws IOException {
        System.load("/home/edward/deliverence/native/target/native-lib-only/libdeliverance.so");
        TensorCache tc = new TensorCache(new MetricRegistry());
        NativeSimdTensorOperations n = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tc).get());
        String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(n),
                new MetricRegistry(), tc, new KvBufferCacheSettings(true))) {
            String prompt = "What is the best season to plant avocados?";
            PromptContext ctx;
            {
                PromptSupport ps = m.promptSupport().get();
                Tool t = Tool.from(Function.builder().name("hello").build());
                ctx = ps.builder().addSystemMessage("You are a chatbot that writes short correct responses.")
                        .addUserMessage(prompt).build(t);
                String expected = """
                        <|system|>
                        You are a chatbot that writes short correct responses.</s>
                        <|user|>
                        What is the best season to plant avocados?</s>
                        <|assistant|>
                        """;
                assertEquals(expected, ctx.getPrompt());// it does not change the prompt to have tools

                Response r = m.generate(UUID.randomUUID(), ctx, new GeneratorParameters().withSeed(42), (s1, f1) -> {});

                System.out.println(r.responseText);
                /*
                assertEquals("""
                        The best thing to do is to look for the plant that best suits your needs and preferences. Avocados are a popular fruit that are grown in many regions around the world. Some of the best regions for avocado production include California, Mexico, and Peru.
                        """, r.responseText);*/
            }
        }
    }
}
