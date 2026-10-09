package io.teknek.deliverance.integration.qwen;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.generator.Response;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.AutoModelForCausaLm;
import io.teknek.deliverance.model.InferenceProfiler;
import io.teknek.deliverance.model.LocalGenerationBackend;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.safetensors.prompt.PromptContext;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.UUID;

@Tag("large-model")
class AntaresBenchmarkCasesIT {
    @Test
    void rawCompletionBenchmark() {
        boolean previousProfiling = InferenceProfiler.isEnabled();
        InferenceProfiler.setEnabled(true);
        ModelFetcher fetch = new ModelFetcher("edwardcapriolo", "antares-1b-JQ4").withDownload(false);
        Assumptions.assumeTrue(fetch.pathForModel().toFile().isDirectory(),
                "Antares cache is not present: " + fetch.pathForModel());
        int maxTokens = Integer.getInteger("antares.benchmark.maxTokens", 10);
        String promptText = System.getProperty("antares.benchmark.prompt",
                "Analyze this codebase for CWE-78: Improper Neutralization of Special Elements used in an OS Command "
                        + "('OS Command Injection'). The product constructs all or part of an OS command using "
                        + "externally influenced input but does not neutralize special elements. Use the terminal "
                        + "tool to explore the repository, then submit the vulnerable repository-relative file path "
                        + "or declare no vulnerability found.");

        try (AbstractModel model = AutoModelForCausaLm.newBuilder(fetch)
                .withDownload(false)
                .buildLocalTransformerModel()) {
            InferenceProfiler.reset();
            PromptContext prompt = PromptContext.of(promptText);
            Response response = model.generateWithBackendRef(UUID.randomUUID(), prompt,
                    new GeneratorParameters().withTemperature(0.0f).withMaxTokens(maxTokens).withSeed(42),
                    (next, nextRaw, nextCleaned, timing) -> {
                        System.out.print(nextCleaned);
                        System.out.flush();
                    }, new LocalGenerationBackend(model));
            System.out.println();
            double prefillMs = response.timeToFirstTokenMs;
            double decodeMs = Math.max(0.0, response.totalTimeMs - prefillMs);
            long decodeTokens = Math.max(0, response.generatedTokens.size() - 1L);
            double prefillTokensPerSecond = prefillMs == 0.0
                    ? 0.0 : response.promptTokens / (prefillMs / 1000.0);
            double decodeTokensPerSecond = decodeMs == 0.0
                    ? 0.0 : decodeTokens / (decodeMs / 1000.0);
            System.out.printf(java.util.Locale.ROOT,
                    "[antares-benchmark-it] prompt_tokens=%d generated=%d total_ms=%.1f "
                            + "ttft_ms=%.1f prefill_tok_s=%.2f decode_ms=%.1f decode_tok_s=%.2f finish=%s%n",
                    response.promptTokens, response.generatedTokens.size(), response.totalTimeMs,
                    prefillMs, prefillTokensPerSecond, decodeMs, decodeTokensPerSecond, response.finishReason);
            InferenceProfiler.printSummary("antares-benchmark", 30);
        } finally {
            InferenceProfiler.setEnabled(previousProfiling);
        }
    }
}
