package io.teknek.deliverance.safetensors.prompt;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.fetch.ModelFetcher;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class DirectNoPromptTest {

    @Test
    public void noPromptContext() throws IOException {
        String modelName = "microlama-lidor-finetuned";
        String modelOwner = "lidoreliya13";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(), new MetricRegistry())) {
            String prompt = "What comes next in the sequence? 1, 2 ";
            PromptContext ctx = PromptContext.of(prompt);
            Response r = m.generate(UUID.randomUUID(), ctx, new GeneratorParameters().withSeed(43), (s, f1) -> {});
            Assertions.assertEquals("2,33,44,55,66,77,88,99,1010,1111,1212,1313,1414,1515,1616,1717,1818,1919,2020,2121,2222,2322,2422,2522,2622,2722,2822,2922,3022,3122,3222,3322,3422,3522,3622,3722,382", r.responseText);
        }
    }
}
