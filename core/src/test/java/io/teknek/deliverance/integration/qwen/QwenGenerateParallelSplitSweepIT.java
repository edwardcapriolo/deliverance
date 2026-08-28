package io.teknek.deliverance.integration.qwen;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelConfig;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.DoNothingGenerateEvent;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.model.TensorProviderKind;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import io.teknek.deliverance.safetensors.prompt.PromptSupport;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertFalse;

@Tag("longtest")
class QwenGenerateParallelSplitSweepIT {
    private static final int[] SPLITS = {8, 16, 32};
    private static final String PROMPT = "Read the puzzle carefully and answer with a clear explanation. "
            + "A company reserves five parking spaces in order for the CEO, president, vice president, secretary, "
            + "and treasurer. The cars are red, blue, green, yellow, and purple. The first space is red. "
            + "A blue car is between the red car and the green car. The last space is purple. The secretary drives "
            + "yellow. Alice parks next to David. Enid drives green. Bert parks between Cheryl and Enid. "
            + "David parks in the last space. Who is the secretary, and what are the car colors from first to last?";

    @ParameterizedTest(name = "{0} simd={2}")
    @MethodSource("generateCases")
    void generateFixedSimdSplitSweep(String label, ModelRef modelRef, int simdSplit) {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();
        ModelFetcher fetcher = new ModelFetcher(modelRef.owner(), modelRef.model()).withDownload(false);
        Assumptions.assumeTrue(fetcher.pathForModel().toFile().isDirectory(),
                modelRef.model() + " cache is not present: " + fetcher.pathForModel());
        try (AbstractModel model = builder(fetcher, modelRef.configPath(), simdSplit).buildLocalTransformerModel()) {
            PromptContext prompt = promptContext(model.promptSupport());
            int maxTokens = Integer.getInteger("qwen.generate.maxTokens", 128);
            Response response = model.generate(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withTemperature(0.0f).withMaxTokens(maxTokens).withSeed(42),
                    new DoNothingGenerateEvent());
            long decodeTokens = Math.max(0, response.generatedTokens.size() - 1L);
            double decodeMs = Math.max(0.0, response.totalTimeMs - response.timeToFirstTokenMs);
            double decodeTokS = decodeMs == 0.0 ? 0.0 : decodeTokens / (decodeMs / 1000.0);
            System.out.printf(Locale.ROOT,
                    "[qwen-generate-sweep] model=%s split=%d prompt_tokens=%d generated=%d ttft_ms=%.3f decode_ms=%.3f decode_tokens=%d decode_tok_s=%.3f total_ms=%.3f finish=%s provider=%s provider_split=%d%n",
                    modelRef.model(), simdSplit, response.promptTokens, response.generatedTokens.size(),
                    response.timeToFirstTokenMs, decodeMs, decodeTokens, decodeTokS, response.totalTimeMs,
                    response.finishReason, model.getTensorProviderName(), model.getTensorProviderParallelSplitSize());
            assertFalse(response.responseTextWithSpecialTokens.isBlank());
        } finally {
            InferenceProfiler.printSummary("generate-sweep model=" + modelRef.model() + " split=" + simdSplit, 30);
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    private static AutoModelForCausaLm.Builder builder(ModelFetcher fetcher, Optional<String> configPath,
            int simdSplit) {
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(fetcher).withDownload(false);
        configPath.ifPresent(path -> builder.withConfig(AutoModelConfig.fromJson(resolveRepoPath(path))));
        return builder.withParallelSplitSizeFixed(TensorProviderKind.SIMD, simdSplit);
    }

    private static Stream<Arguments> generateCases() {
        Set<String> enabledModels = csvProperty("qwen.generate.models");
        Set<String> enabledSplits = csvProperty("qwen.generate.splits");
        return Stream.of(
                        new ModelRef("qwen4b", "edwardcapriolo", "Qwen3-4B-JQ4",
                                Optional.of("benchmarks/configs/qwen3-4b-jq4.json")),
                        new ModelRef("qwen06b", "edwardcapriolo", "Qwen3-0.6B-JQ4",
                                Optional.of("benchmarks/configs/qwen3-0.6b-jq4.json")))
                .filter(model -> enabledModels.isEmpty() || enabledModels.contains(model.label()))
                .flatMap(model -> IntStream.of(SPLITS)
                        .filter(split -> enabledSplits.isEmpty() || enabledSplits.contains(Integer.toString(split)))
                        .mapToObj(split -> Arguments.of(model.label(), model, split)));
    }

    private static PromptContext promptContext(Optional<PromptSupport> promptSupport) {
        if (promptSupport.isPresent()) {
            return promptSupport.get().builder().addUserMessage(PROMPT).build();
        }
        return PromptContext.of(PROMPT);
    }

    private static Set<String> csvProperty(String name) {
        String value = System.getProperty(name, "").trim();
        if (value.isEmpty()) {
            return Set.of();
        }
        return Stream.of(value.split(","))
                .map(String::trim)
                .filter(part -> !part.isEmpty())
                .collect(java.util.stream.Collectors.toSet());
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
        throw new IllegalStateException("Unable to locate " + relativePath + " from " + Path.of("").toAbsolutePath());
    }

    private record ModelRef(String label, String owner, String model, Optional<String> configPath) {
    }
}
