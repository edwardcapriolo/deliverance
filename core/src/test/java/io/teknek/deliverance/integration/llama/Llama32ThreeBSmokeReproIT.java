package io.teknek.deliverance.integration.llama;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.math.WrappedForkJoinPool;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.CausalLanguageModel;
import io.teknek.deliverance.model.DefaultCausalLanguageModel;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import org.junit.jupiter.api.Test;

import java.util.UUID;
import java.util.concurrent.ForkJoinPool;
import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertFalse;

/** Exact reproduction of the smoke benchmark's Llama case, including warmup. */
public class Llama32ThreeBSmokeReproIT {
    private static final String PUZZLE = "Read the puzzle carefully and answer with a clear explanation. "
            + "A company reserves five parking spaces in order for the CEO, president, vice president, secretary, "
            + "and treasurer. The cars are red, blue, green, yellow, and purple. The first space is red. "
            + "A blue car is between the red car and the green car. The last space is purple. "
            + "The secretary drives yellow. Alice parks next to David. Enid drives green. "
            + "Bert parks between Cheryl and Enid. David parks in the last space. "
            + "Who is the secretary, and what are the car colors from first to last?";

    @Test
    void smokeCaseWarmupThenMeasuredGeneration() throws Exception {
        KvBufferCacheSettings settings = new KvBufferCacheSettings(true)
                .withMaxEntries(10_000)
                .withBlockSize(32)
                .withMaxPrefixTokensPerPrompt(512)
                .withContextRowsPerPageTarget(32);
        WrappedForkJoinPool pool = new WrappedForkJoinPool(new ForkJoinPool(16));
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(
                        new ModelFetcher("tjake", "Llama-3.2-3B-Instruct-JQ4"))
                .withWorkingQuantType(DType.I8)
                .withOutputHeadQuantization(DType.Q4)
                .withKvBufferCacheSettings(settings)
                .withWrappedForkJoinPool(pool);

        Path tracePath = Path.of("target/llama-smoke-trace.jsonl");
        try {
            Files.createDirectories(tracePath.getParent());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        try (CausalLanguageModel model = builder.build();
             BufferedWriter trace = Files.newBufferedWriter(tracePath)) {
            AbstractModel local = ((DefaultCausalLanguageModel) model).localTransformerModel();
            AtomicReference<String> run = new AtomicReference<>("warmup");
            local.setLayerDebugHook(event -> writeTrace(trace, run.get(), event));
            PromptContext prompt = model.promptSupport().orElseThrow().builder()
                    .addUserMessage(PUZZLE)
                    .build();
            GeneratorParameters parameters = new GeneratorParameters()
                    .withTemperature(0.0f)
                    .withMaxTokens(64)
                    .withSeed(42);

            Response warmup = model.generate(UUID.randomUUID(), prompt, parameters, new DoNothingGenerateEvent());
            System.out.println("LLAMA_SMOKE_WARMUP=" + warmup.responseText.replace("\n", "\\n"));
            run.set("measured");
            Response measured = model.generate(UUID.randomUUID(), prompt, parameters, new DoNothingGenerateEvent());
            System.out.println("LLAMA_SMOKE_MEASURED=" + measured.responseText.replace("\n", "\\n"));

            assertFalse(measured.responseText.isBlank());
            local.clearLayerDebugHook();
        }
        System.out.println("LLAMA_SMOKE_TRACE=" + tracePath);
    }

    private static void writeTrace(BufferedWriter trace, String run, AbstractModel.LayerDebugEvent event) {
        try {
            AbstractTensor tensor = event.hiddenStates();
            Map<String, Object> record = new LinkedHashMap<>();
            record.put("run", run);
            record.put("layer", event.layerIndex());
            record.put("stage", event.stage());
            record.put("shape", tensor.shape().shapeArray());
            ArrayList<Float> sample = new ArrayList<>();
            int limit = Math.min(32, Math.toIntExact(tensor.shape().size()));
            for (int index = 0; index < limit; index++) {
                sample.add(tensor.get(index / tensor.shape().last(), index % tensor.shape().last()));
            }
            record.put("sample", sample);
            synchronized (trace) {
                trace.write(JsonUtils.om.writeValueAsString(record));
                trace.newLine();
                trace.flush();
            }
        } catch (Exception e) {
            throw new RuntimeException("Unable to write Llama smoke trace", e);
        }
    }
}
