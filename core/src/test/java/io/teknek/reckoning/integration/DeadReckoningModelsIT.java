package io.teknek.reckoning.integration;

import io.teknek.reckoning.generator.GeneratorParameters;
import io.teknek.reckoning.generator.Response;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.AutoModelConfig;
import io.teknek.reckoning.model.AutoModelForCausaLm;
import io.teknek.reckoning.model.DeadReckoningParams;
import io.teknek.reckoning.model.DeadReckoningSelector;
import io.teknek.reckoning.model.DeadReckoningTraceCheck;
import io.teknek.reckoning.model.DoNothingGenerateEvent;
import io.teknek.reckoning.safetensors.fetch.ModelFetcher;
import io.teknek.reckoning.safetensors.prompt.PromptContext;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.net.InetAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.EnumSet;
import java.util.List;
import java.util.UUID;

/** Fast six-token Dead Reckoning traces for the supported small model families. */
@Tag("large-model")
public class DeadReckoningModelsIT {
    private static final String PROMPT = "generate the first three letters of the alphabet.";

    @Test
    void qwen06bDense() throws Exception {
        traceAndCompare("qwen06b-dense", "Qwen", "Qwen3-0.6B", null);
    }

    @Test
    void qwen06bJq4() throws Exception {
        traceAndCompare("qwen06b-jq4", "edwardcapriolo", "Qwen3-0.6B-JQ4", null);
    }

    @Test
    void tinyLlamaDeliverance() throws Exception {
        traceAndCompare("tinyllama-deliverance", "tjake", "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4", null);
    }

    @Test
    void tinyLlamaJq4() throws Exception {
        traceAndCompare("tinyllama-jq4", "tjake", "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4", null);
    }

    @Test
    void gpt2() throws Exception {
        traceAndCompare("gpt2", "openai-community", "gpt2", null,
                "The first three letters of the alphabet are");
    }

    @Test
    void gemma2Jq4() throws Exception {
        traceAndLog("gemma2-jq4", "tjake", "gemma-2-2b-it-JQ4", null);
    }

    /*
    @Test
    @Tag("large-model")
    void gemma4E2bJq4() throws Exception {
        traceAndCompare("gemma4-e2b", "google", "gemma-4-E2B-it", null);
    }
    */

    @Test
    void nemotron3bArJq4() throws Exception {
        traceAndCompare("nemotron3b-ar-jq4", "edwardcapriolo", "Nemotron-Labs-Diffusion-3B-JQ4",
                "benchmarks/configs/nemotron-labs-diffusion-3b-base-jq4-ar.json");
    }

    @Test
    void nemotron3bDiffusionJq4() throws Exception {
        traceAndCompare("nemotron3b-diffusion-jq4", "edwardcapriolo", "Nemotron-Labs-Diffusion-3B-JQ4",
                "benchmarks/configs/nemotron-labs-diffusion-3b-base-jq4-diffusion.json");
    }

    @Test
    void antares1bJq4() throws Exception {
        traceAndLog("antares1b-jq4", "edwardcapriolo", "antares-1b-JQ4", null);
    }

    private static void traceAndCompare(String traceName, String owner, String modelName, String configPath)
            throws Exception {
        Path current = traceAndLog(traceName, owner, modelName, configPath);
        compareWithPrevious(current, modelName);
    }

    private static void traceAndCompare(String traceName, String owner, String modelName, String configPath,
            String promptText) throws Exception {
        Path current = traceAndLog(traceName, owner, modelName, configPath, promptText);
        compareWithPrevious(current, modelName);
    }

    private static void compareWithPrevious(Path current, String modelName) {
        if (!DeadReckoningTraceCheck.sameAsPrevious(current)) {
            throw new AssertionError("Dead Reckoning trace changed for " + modelName);
        }
    }

    private static Path traceAndLog(String traceName, String owner, String modelName, String configPath)
            throws Exception {
        return traceAndLog(traceName, owner, modelName, configPath, PROMPT);
    }

    private static Path traceAndLog(String traceName, String owner, String modelName, String configPath,
            String promptText) throws Exception {
        ModelFetcher fetcher = new ModelFetcher(owner, modelName).withDownload(true);
        fetcher.maybeDownload();

        String host = InetAddress.getLocalHost().getHostName();
        Path output = Path.of("dead-reckoning", host,
                traceName + "-" + Instant.now().toEpochMilli() + ".jsonl");
        DeadReckoningSelector selector = DeadReckoningSelector.regex(null, null, null, null,
                DeadReckoningSelector.Capture.STATS, 32,
                EnumSet.allOf(DeadReckoningSelector.Statistic.class));
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(fetcher)
                .withDeadReckoning(new DeadReckoningParams(output, 32, List.of(selector)));
        if (configPath != null) {
            builder.withConfig(AutoModelConfig.fromJson(resolveRepoPath(configPath)));
        }
        try (AbstractModel model = builder.buildLocalTransformerModel()) {
            PromptContext prompt = model.promptSupport()
                    .map(support -> support.builder().addUserMessage(promptText).build())
                    .orElseGet(() -> PromptContext.of(promptText));
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters()
                            .withTemperature(0.0f)
                            .withSeed(42)
                            .withMaxTokens(6)
                            .withLogProbs(true)
                            .withTimeout(Duration.ofSeconds(30))
                            .withTopLogProbs(10),
                    new DoNothingGenerateEvent());
            System.out.println("DEAD_RECKONING_RESPONSE model=" + modelName
                    + " text=" + response.responseTextWithSpecialTokens.replace("\n", "\\n")
                    + " tokens=" + response.generatedTokens
                    + " finish=" + response.finishReason);
        }
        System.out.println("DEAD_RECKONING_TRACE=" + output.toAbsolutePath());
        return output;
    }

    private static Path resolveRepoPath(String relativePath) {
        Path current = Path.of("").toAbsolutePath();
        while (current != null) {
            Path candidate = current.resolve(relativePath);
            if (Files.exists(candidate)) {
                return candidate;
            }
            current = current.getParent();
        }
        throw new IllegalStateException("Unable to locate " + relativePath);
    }
}
