package io.teknek.deliverance.integration.qwen;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelConfig;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.model.TensorProviderKind;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
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
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;

@Tag("longtest")
class QwenPrefillParallelSplitSweepIT {
    private static final int[] SPLITS = {4, 8, 16, 32, 64};
    private static final String PROMPT = "Read the puzzle carefully and answer with a clear explanation. "
            + "A company reserves five parking spaces in order for the CEO, president, vice president, secretary, "
            + "and treasurer. The cars are red, blue, green, yellow, and purple. The first space is red. "
            + "A blue car is between the red car and the green car. The last space is purple. The secretary drives "
            + "yellow. Alice parks next to David. Enid drives green. Bert parks between Cheryl and Enid. "
            + "David parks in the last space. Who is the secretary, and what are the car colors from first to last?";

    @ParameterizedTest(name = "{0} simd={2}")
    @MethodSource("prefillCases")
    void prefillOnlyFixedSimdSplitSweep(String label, ModelRef modelRef, int simdSplit) {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        InferenceProfiler.reset();
        ModelFetcher fetcher = new ModelFetcher(modelRef.owner(), modelRef.model()).withDownload(false);
        Assumptions.assumeTrue(fetcher.pathForModel().toFile().isDirectory(),
                modelRef.model() + " cache is not present: " + fetcher.pathForModel());
        try (AbstractModel model = builder(fetcher, modelRef.configPath(), simdSplit).buildLocalTransformerModel()) {
            int[] promptTokens = model.constructPromptTokensForRuntime(PROMPT);
            long start = System.nanoTime();
            try (KvBufferCache.KvBuffer kv = model.newKvBuffer();
                 AbstractTensor prefill = model.batchForward(promptTokens, 0, kv)) {
                long elapsedNanos = System.nanoTime() - start;
                double elapsedMs = elapsedNanos / 1_000_000.0;
                double tokensPerSecond = promptTokens.length / (elapsedNanos / 1_000_000_000.0);
                System.out.printf(Locale.ROOT,
                        "[qwen-prefill-sweep] model=%s split=%d prompt_tokens=%d prefill_ms=%.3f prefill_tok_s=%.3f output_shape=%s provider=%s provider_split=%d%n",
                        modelRef.model(), simdSplit, promptTokens.length, elapsedMs, tokensPerSecond, prefill.shape(),
                        model.getTensorProviderName(), model.getTensorProviderParallelSplitSize());
                assertEquals(promptTokens.length, prefill.shape().first());
            }
        } finally {
            InferenceProfiler.printSummary("prefill-sweep model=" + modelRef.model() + " split=" + simdSplit, 20);
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }

    private static AutoModelForCausaLm.Builder builder(ModelFetcher fetcher, Optional<String> configPath,
            int simdSplit) {
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(fetcher).withDownload(false);
        configPath.ifPresent(path -> builder.withConfig(AutoModelConfig.fromJson(resolveRepoPath(path))));
        return builder.withParallelSplitSizeFixed(TensorProviderKind.SIMD, simdSplit);
    }

    private static Stream<Arguments> prefillCases() {
        Set<String> enabledModels = csvProperty("qwen.prefill.models");
        Set<String> enabledSplits = csvProperty("qwen.prefill.splits");
        return Stream.of(
                        new ModelRef("qwen06b", "edwardcapriolo", "Qwen3-0.6B-JQ4",
                                Optional.of("benchmarks/configs/qwen3-0.6b-jq4.json")),
                        new ModelRef("qwen4b", "edwardcapriolo", "Qwen3-4B-JQ4",
                                Optional.of("benchmarks/configs/qwen3-4b-jq4.json")))
                .filter(model -> enabledModels.isEmpty() || enabledModels.contains(model.label()))
                .flatMap(model -> IntStream.of(SPLITS)
                        .filter(split -> enabledSplits.isEmpty() || enabledSplits.contains(Integer.toString(split)))
                        .mapToObj(split -> Arguments.of(model.label(), model, split)));
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
