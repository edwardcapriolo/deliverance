import com.codahale.metrics.MetricRegistry;
import com.fasterxml.jackson.annotation.JsonCreator;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.model.ModelSupport;
import io.teknek.deliverance.model.qwen2.Qwen2ModelType;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.*;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;
import java.util.*;

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
                new MetricRegistry(), tensorCache, new KvBufferCacheSettings(true), fetch)) {
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

            Response k = m.generate(uuid, g.build(), new GeneratorParameters().withTemperature(0.0f),
                    new DoNothingGenerateEvent());
            assertTrue(k.responseText.contains("New York City"));

        }
    }

    @Test
    public void toolTest() throws IOException {
        File soFile = new File("target/native-lib-only/libdeliverance.so");
        assertTrue(soFile.exists());
        System.load(soFile.getAbsolutePath());

        //ModelFetcher fetch = new ModelFetcher("tjake", "Qwen2.5-0.5B-Instruct-JQ4");
        ModelFetcher fetch = new ModelFetcher("tjake", "Llama-3.1-8B-Instruct-JQ4");
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache).get());
        String text = new Scanner(ModelSupport.class
                .getResourceAsStream("/llama_tool_fix.jinja"), "UTF-8").useDelimiter("\\A").next();
        Tool tool = Tool.from(Function.builder().name("flip_coin").description("This methods will flip a coin. The result will be H for heads or T for tails.").build());
        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),
                new MetricRegistry(), tensorCache, new KvBufferCacheSettings(true), fetch)) {
            String prompt = "Call the function flip_coin print the result.";
            PromptSupport.Builder g = m.promptSupport().get().builder()
                    .useSpecifiedTemplate(text)
                    .addToolCall(new ToolCall("flip_coin", "flip_coin1", Map.of()))
                    .addUserMessage(prompt);

            PromptContext c = g.build(tool);
            System.out.println(c.getPrompt());
            /*
            Assertions.assertEquals("""
template:<|start_header_id|>system<|end_header_id|>

Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.Do not use variables.

{"type": "function", "function": {"name": "coinflip", "description": "Flip a two sided coin", "parameters": {"type": "object", "properties": {}, "required": []}}}

<|eot_id|><|start_header_id|>user<|end_header_id|>

Use the coinflip tool any analyze the result<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n""", g.build(tool).getPrompt());*/
           // var uuid = UUID.randomUUID();

           // Response k = m.generate(uuid, c, new GeneratorParameters().withTemperature(0.1f),
           //         (int next, String nextRaw, String nextCleaned, float timing) -> {
           //             System.out.println(" "+ nextCleaned +" ");
           //         });

           // System.out.println(k);
           // assertTrue(k.responseText.contains("flip_coin"));
            // comes out like this {responseText=' {"name": "flip_coin", "parameters": {}}
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
                new MetricRegistry(), tensorCache, new KvBufferCacheSettings(true), fetch)) {
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
