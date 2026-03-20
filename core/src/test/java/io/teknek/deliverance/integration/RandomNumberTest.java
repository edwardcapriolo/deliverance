package io.teknek.deliverance.integration;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.*;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.TensorCache;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.List;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class RandomNumberTest {

    private void loadNative(){
        File findLib = new File("target/native-lib-only/libdeliverance.so");
        if (findLib.exists()) {
            System.load(findLib.getAbsolutePath());
        }
    }
    @Test
    public void sample() {
        loadNative();
        String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache).get());

        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),
                new MetricRegistry(), tensorCache, new KvBufferCacheSettings(true), fetch, new TokenizerRenderer(), new DefaultToolCallParser())) {
            String prompt = "Pick a random number between 0 and 100";
            PromptContext ctx = PromptContext.of(prompt);
            var uuid = UUID.randomUUID();

            Response k = m.generate(uuid, ctx, new GeneratorParameters().withTemperature(0.0f).withSeed(99999),
                    new DoNothingGenerateEvent());
            System.out.println(k);
            assertEquals("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", k.responseText);
        }
    }


    @Test
    public void calc() {
        String modelName = "Llama-3.2-3B-Instruct-JQ4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        var uuid = UUID.randomUUID();
        /*loadNative();

        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        TensorCache tensorCache = new TensorCache(mr);
        NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(tensorCache).get());

        try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),
               mr, tensorCache, new KvBufferCacheSettings(true), fetch, new TokenizerRenderer())) {*/
        try (AbstractModel m = AutoModelForCausaLm.newBuilder(fetch).withWorkingQuantType(DType.I8)
                .withTokenTokenRenderer(new TokenizerRenderer()).build()){
            String prompt = "Generate a java interface named Shape with a method name calculateArea.";
            PromptContext ctx = m.promptSupport().get().builder()
                    .addSystemMessage("You are an assistant that produces concise, production-grade software.")
                    .addSystemMessage("Output java code.")
                    .addSystemMessage("Refrain from editorializing your reply.")
                    .addSystemMessage("Generate java code into the package 'io.teknek.shape' .")
                    .addSystemMessage("Do not import java.awt")
                    //.addSystemMessage("You are a direct, concise AI. Do not provide explanations, justifications, or conversational filler. Only output the final answer.")
                    .addUserMessage("Generate a java interface named Shape with a method named area that returns a double.")
                    .addUserMessage("Generate a java class named Circle that extends the Shape interface.")
                    .build();


            Response k = m.generate(uuid, ctx, new GeneratorParameters()
                    .withNtokens(512)
                            .withIncludeStopStrInOutput(false)
                    .withStopWords(List.of("<|eot_id|>"))
                    .withTemperature(0.2f).withSeed(99998), new DoNothingGenerateEvent());

            assertEquals("""
Here is the Java code that meets the specifications:

```java
package io.teknek.shape;

public interface Shape {    
    double area();
}

public class Circle extends Shape {
    private double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    @Override
    public double area() {
        return Math.PI * radius * radius;
    }
}
```

This code defines a `Shape` interface with a single method `area()`, which returns a `double` value. It also defines a `Circle` class that extends the `Shape` interface and implements the `area()` method. The `Circle` class has a private `radius` field and a constructor that takes a `radius` value. The `area()` method returns the area of the circle using the formula `ÏĢr^2`.
""".trim(), k.responseText);
        }
    }

    @Test
    public void mdCleanup(){
        String in = """
                THis is the way you should code:
                ```java
                public int x(){
                return 3;
                }
                ```
                That was great right?
                """;
        Assertions.assertEquals("""
public int x(){
return 3;
}
                """, processResponse(in));
    }

    public static String processResponse(String input){
        String s = "```java";
        int indexOfStart = input.indexOf(s);
        if (indexOfStart == -1){
            //assume not MD all code
            return input;
        }
        int end = input.lastIndexOf("```");
        if (end == -1){
            end = input.length() -1;
        }
        return input.substring(indexOfStart + s.length() + 1, end );
    }
}
