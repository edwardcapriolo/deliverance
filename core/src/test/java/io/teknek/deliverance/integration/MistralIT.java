package io.teknek.deliverance.integration;

import com.codahale.metrics.MetricRegistry;
import io.teknek.deliverance.generator.FinishReason;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.GenerateEvent;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.*;
import io.teknek.deliverance.tensor.ArrayQueueTensorAllocator;
import io.teknek.deliverance.tensor.operations.ConfigurableTensorProvider;
import io.teknek.deliverance.toolcallparser.LlamaToolCallParser;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class MistralIT {
    @Test
    public void chat(){
        ModelFetcher fetch = new ModelFetcher("tjake", "Mistral-7B-Instruct-v0.3-JQ4");

        try (WrappedForkJoinPool pool = new WrappedForkJoinPool(WrappedForkJoinPool.autoSizeByCores());
             AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withTensorProvider(new ConfigurableTensorProvider(new ArrayQueueTensorAllocator(new MetricRegistry()), pool)).build()) {
            String prompt = "Who is Edward Capriolo";
            PromptSupport.Builder g = model.promptSupport().get().builder()
                    .addUserMessage(prompt);
            var uuid = UUID.randomUUID();

            Response response = model.generate(uuid, g.build(), new GeneratorParameters()
                            .withTemperature(0.0f).withNtokens(500).withMaxTokens(150),
                    new GenerateEvent() {
                        @Override
                        public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                            System.out.println(nextCleaned);
                        }
                    });
            //assertEquals("", response.responseText);
        }
    }

    @Disabled
    public void completeToolWithMockAgent(){
        ModelFetcher fetch = new ModelFetcher("tjake", "Mistral-7B-Instruct-v0.3-JQ4");

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch).withToolCallParser(new LlamaToolCallParser()).build()) {
            String prompt = "What is the temperature in NYC right now?";
            PromptSupport.Builder builder = model.promptSupport().get().builder()
                    .addUserMessage(prompt);
            builder.addGenerationPrompt(true);
            var uuid = UUID.randomUUID();

            Tool t = Tool.from(
                    Function.builder()
                            .name("get_current_temperature")
                            .description("Simulates getting the current temperature at a location.")
                            .addParameter("location", "string", "The location to get the temperature for, in the format \"City, Country\".", true)
                            .addParameter("unit", "string", "The unit to return the temperature in (e.g., \"celsius\", \"fahrenheit\").", true)
                            .build()
            );
            builder.addToolItem(t);
            Assertions.assertEquals("""
                    PromptContext{prompt='[AVAILABLE_TOOLS] [{"type": "function", "function": {"name": "get_current_temperature", "description": "Simulates getting the current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, Country\\"."}, "unit": {"type": "string", "description": "The unit to return the temperature in (e.g., \\"celsius\\", \\"fahrenheit\\")."}}, "required": ["location", "unit"]}}}, {"type": "function", "function": {"name": "get_current_temperature", "description": "Simulates getting the current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, Country\\"."}, "unit": {"type": "string", "description": "The unit to return the temperature in (e.g., \\"celsius\\", \\"fahrenheit\\")."}}, "required": ["location", "unit"]}}}][/AVAILABLE_TOOLS][INST] What is the temperature in NYC right now?[/INST]'}""", builder.build().toString());

            GeneratorParameters p = new GeneratorParameters()
                    .withTemperature(0.0f).withNtokens(8000).withMaxTokens(300);
            GenerateEvent e = new GenerateEvent() {
                @Override
                public void emit(int next, String nextRaw, String nextCleaned, float timing) {
                    System.out.println(nextCleaned);
                }
            };
            Response response = model.generate(uuid, builder.build(), p, e);
            System.out.println(response.responseText);
            Assertions.assertEquals(FinishReason.TOOL_CALLS, response.finishReason);
            Assertions.assertEquals(1, response.toolCalls.size());

            ToolCall f = response.toolCalls.get(0);
            System.out.println("Agent is calling tool " + f );
            //assertEquals("", response.responseText);

            builder.addToolCall(f);
            builder.addToolResult(ToolResult.from(f.getName(), null, 20f));

            System.out.println("Second prompt " + builder.build());
            Response r2 = model.generate(UUID.randomUUID(), builder.build(), p, e);
            System.out.println(r2.responseText);

            //Assertions.assertTrue(r2.responseText, r2.responseText.contains("20"));
            //logger.info("Response: {}", r2.responseText);
        }
    }
}
