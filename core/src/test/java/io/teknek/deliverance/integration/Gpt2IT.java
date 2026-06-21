package io.teknek.deliverance.integration;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.*;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class Gpt2IT {
    @Tag("large-model")
    @Test
    public void chat(){
        ModelFetcher fetch = new ModelFetcher("openai-community", "gpt2-large");
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
             AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withTensorProvider(new ConfigurableTensorProvider(new ArrayQueueTensorAllocator(new MetricRegistry()), pool)).buildLocalTransformerModel()) {
            String prompt = "Who is Micheal Jordan?";
            //This model does not have prompt support
            var uuid = UUID.randomUUID();
            PromptContext ctx = PromptContext.of(prompt);

            Response response = model.generate(uuid, ctx, new GeneratorParameters()
                            .withTemperature(0.0f).withNtokens(500).withMaxTokens(150),
                  new DoNothingGenerateEvent());
            //TODO cleanup
            System.out.println(response.responseText);
            //assertEquals("", response.responseText);
        }
    }
}
