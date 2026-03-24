package io.teknek.deliverance.integration;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.*;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class Gpt2IT {
    @Test
    public void chat(){
        ModelFetcher fetch = new ModelFetcher("openai-community", "gpt2-large");

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withTensorProvider(new ConfigurableTensorProvider(new TensorCache(new MetricRegistry()))).build()) {
            String prompt = "Who is Micheal Jordan?";
            //This model does not have prompt support
            var uuid = UUID.randomUUID();
            PromptContext ctx = PromptContext.of(prompt);
            Response response = model.generate(uuid, ctx, new GeneratorParameters()
                            .withTemperature(0.0f).withNtokens(500).withMaxTokens(150),
                    new GenerateEvent() {
                        @Override
                        public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                            System.out.println(nextCleaned);
                        }
                    });
            assertEquals("", response.responseText);
        }
    }
}
