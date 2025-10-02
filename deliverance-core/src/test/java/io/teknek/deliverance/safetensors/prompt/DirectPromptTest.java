package io.teknek.deliverance.safetensors.prompt;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.fetch.ModelFetcher;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
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

                Response r = m.generate(UUID.randomUUID(), ctx, 0.0f, 256, (s1, f1) -> {});
                //assertEquals("yo", r.responseText);
            }
            /*
            {
                PromptSupport ps = m.promptSupport().get();
                long start = System.currentTimeMillis();
                ctx = ps.builder().addSystemMessage("You are a math expert.")
                        .addUserMessage("What is the resul of 1 + 1?").build();


                Response r = m.generate(UUID.randomUUID(), ctx, 0.0f, 256, (s1, f1) -> {});
                System.out.println("It took " + (System.currentTimeMillis()-start));
            }

            {
                PromptSupport ps = m.promptSupport().get();
                long start = System.currentTimeMillis();
                ctx = ps.builder().addSystemMessage("You are a math expert.")
                        .addUserMessage("What is the resul of 1 + 1?").build();


                Response r = m.generate(UUID.randomUUID(), ctx, 0.0f, 256, (s1, f1) -> {});
                System.out.println("It took " + (System.currentTimeMillis()-start));
            }*/
        }



        // Generates a response to the prompt and prints it
        // The api allows for streaming or non-streaming responses
        // The response is generated with a temperature of 0.7 and a max token length of 256

        //Generator.Response r = m.generate(UUID.randomUUID(), ctx, 0.0f, 256, (s, f) -> {});
        //System.out.println(r.responseText);
    }
}
