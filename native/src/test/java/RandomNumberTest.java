import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class RandomNumberTest {

    @Test
    public void sample() throws IOException {
        String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache).get());
        ConfigurableTensorProvider withoutNative = new ConfigurableTensorProvider(tensorCache);
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),
                new MetricRegistry(), tensorCache, new KvBufferCacheSettings(true))) {
            String prompt = "Pick a random number between 0 and 100";
            PromptContext ctx = PromptContext.of(prompt);
            var uuid = UUID.randomUUID();

            Response k = m.generate(uuid, ctx, new GeneratorParameters().withTemperature(0.0f).withSeed(99999),(s1, f1) -> {});
            System.out.println(k);
            assertEquals("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", k.responseText);
        }
    }

}
