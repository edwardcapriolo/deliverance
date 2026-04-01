package io.teknek.deliverance.integration;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.UUID;

public class Gemma3IT {
    @Disabled
    public void chat(){
        ModelFetcher fetch = new ModelFetcher("google", "gemma-3-1b-it");

        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
             AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withTensorProvider(new ConfigurableTensorProvider(new TensorCache(new MetricRegistry()), pool)).build()) {
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

            Response k = model.generate(uuid, g.build(), new GeneratorParameters().withTemperature(0.0f).withNtokens(500),
                    new GenerateEvent() {
                        @Override
                        public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                            System.out.println(nextCleaned);
                        }
                    });
        }
    }
}
