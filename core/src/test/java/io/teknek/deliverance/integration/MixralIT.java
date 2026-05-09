package io.teknek.deliverance.integration;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;


import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;

import com.codahale.metrics.ConsoleReporter;
import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;


import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MixralIT {
    @Tag("large-model")
    @Test
    public void chat() {
        ModelFetcher fetch = new ModelFetcher("tjake", "Mixtral-8x7B-Instruct-v0.1-JQ4");

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).build()) {
/*
MetricRegistry registry = new MetricRegistry();

        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
             AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)                                           
                .withMetricRegistry(registry)                                                          
                .withTensorProvider(new ConfigurableTensorProvider(new ArrayQueueTensorAllocator(registry), pool)).build()) {
*/


            boolean doAssert = true;

            {
                String prompt = "What colors are in a rainbow?";
                PromptSupport.Builder g = model.promptSupport().get().builder()
                        .addUserMessage(prompt);

                var uuid = UUID.randomUUID();

                long start = System.currentTimeMillis();
                Response response = model.generate(uuid, g.build(), new GeneratorParameters()
                                .withNtokens(500).withMaxTokens(50).withTopP(0.05f),
                        new GenerateEvent() {
                            @Override
                            public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                                System.out.println(nextCleaned);
                            }
                        });
                System.out.println(System.currentTimeMillis() - start);
                if (doAssert) {
                    assertEquals("""
                            A rainbow consists of several colors, typically appearing in the order of red, orange, yellow, green, blue, indigo, and violet. This sequence is often remembered by the acronym "ROYGBIV." However, it""".trim(), response.responseText.trim());
                } else {
                    System.out.println(response.responseText);
                }
            }

            {
                String prompt = "Create a class in Java with a method that converts fahrenheit to celsius. Only show code.";
                PromptSupport.Builder g = model.promptSupport().get().builder()
                        .addUserMessage(prompt);

                var uuid = UUID.randomUUID();

                long start = System.currentTimeMillis();
                Response response = model.generate(uuid, g.build(), new GeneratorParameters()
                                .withNtokens(500).withMaxTokens(50).withTopP(0.05f),
                        new GenerateEvent() {
                            @Override
                            public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                                System.out.println(nextCleaned);
                            }
                        });
                System.out.println(System.currentTimeMillis() - start);
                if (doAssert) {
/*
            assertEquals("""
 Here's a simple Java class with a method that converts Fahrenheit to Celsius:

```java
public class TemperatureConverter {
    
    public double convertFahrenheitToCelsius(double""".trim(), response.responseText.trim());*/
                }
            }

            {
                String prompt = "Name 3 states in the United States of America.";
                PromptSupport.Builder g = model.promptSupport().get().builder()
                        .addUserMessage(prompt);

                var uuid = UUID.randomUUID();
                long start = System.currentTimeMillis();
                Response response = model.generate(uuid, g.build(), new GeneratorParameters()
                                .withNtokens(500).withMaxTokens(50).withTopP(0.05f),
                        new GenerateEvent() {
                            @Override
                            public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                                System.out.println(nextCleaned);
                            }
                        });
                System.out.println(System.currentTimeMillis() - start);
                if (doAssert) {
                    assertEquals("""                                                                                                     
                            It seems like you're asking for the three states in the United States of America that begin with the letter "S." Here they are:
                            
                            1. California - Officially known as the "Golden State," California is the most""".trim(), response.responseText.trim());
                }

            }


            {
                String prompt = "True or false? Batman wears a mask.";
                PromptSupport.Builder g = model.promptSupport().get().builder()
                        .addUserMessage(prompt);

                var uuid = UUID.randomUUID();
                long start = System.currentTimeMillis();
                Response response = model.generate(uuid, g.build(), new GeneratorParameters()
                                .withNtokens(500).withMaxTokens(70).withTopP(0.05f),
                        new GenerateEvent() {
                            @Override
                            public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                                System.out.println(nextCleaned);
                            }
                        });
                System.out.println(System.currentTimeMillis() - start);
                assertEquals("""
                        True. Batman is well-known for wearing a mask as part of his costume, which is a critical aspect of his secret identity as Bruce Wayne. This mask usually covers the upper part of his face, including his eyes and nose, while leaving his mouth uncovered. The mask is often depicted as being made of durable materials like kevlar""".trim(), response.responseText.trim());
            }


        }
    }

}
