package io.teknek.deliverance.safetensors.prompt;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.fetch.ModelFetcher;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class DirectPromptTest {

    @Test
    public void sample() throws IOException {
        String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(), new MetricRegistry())) {

            String prompt = "What is the best season to plant avocados?";
            PromptContext ctx;
            {
                PromptSupport ps = m.promptSupport().get();
                ctx = ps.builder().addSystemMessage("You are a chatbot that writes short correct responses.")
                        .addUserMessage(prompt).build();
                String expected = """
                        <|system|>
                        You are a chatbot that writes short correct responses.</s>
                        <|user|>
                        What is the best season to plant avocados?</s>
                        <|assistant|>
                        """;
                assertEquals(expected, ctx.getPrompt());
            }
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

                Response r = m.generate(UUID.randomUUID(), ctx, 0.0f, 256, Optional.of(42), (s1, f1) -> {});
                System.out.println(r.responseText);
            }
        }
    }
}
