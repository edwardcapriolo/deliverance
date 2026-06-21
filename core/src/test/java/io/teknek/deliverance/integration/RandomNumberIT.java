package io.teknek.deliverance.integration;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.*;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.tensor.operations.NativeSimdTensorOperations;
import io.teknek.deliverance.toolcallparser.DefaultToolCallParser;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class RandomNumberIT {


    @Test
    public void sample() {
        String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        MetricRegistry mr = new MetricRegistry();
        ArrayQueueTensorAllocator arrayQueueTensorAllocator = new ArrayQueueTensorAllocator(mr);
        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores())) {
            NativeSimdTensorOperations operation = new NativeSimdTensorOperations(new ConfigurableTensorProvider(arrayQueueTensorAllocator, pool).get());

            try (AbstractModel m = ModelSupport.loadModel(f, DType.F32, DType.I8, new ConfigurableTensorProvider(operation),
                    new MetricRegistry(), arrayQueueTensorAllocator, new KvBufferCacheSettings(true), fetch,
                    new DefaultToolCallParser(), pool)) {
                String prompt = "Pick a random number between 0 and 100";
                PromptContext ctx = m.promptSupport().get().builder()
                        .addUserMessage(prompt)
                        .build();
                var uuid = UUID.randomUUID();
                Response k = m.generate(uuid, ctx, new GeneratorParameters().withTemperature(0.0f).withSeed(99999)
                                .withMaxTokens(4),
                        new DoNothingGenerateEvent());
                System.out.println(k);
                assertEquals(":\n\n1", k.responseText);
            }
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
