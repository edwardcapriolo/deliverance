package io.teknek.deliverance.integration.qwen;

import org.junit.jupiter.api.Tag;

import java.util.Optional;

/** Baseline Qwen4B profiler run using the known-good non-I8-KV configuration. */
@Tag("large-model")
class Qwen4bNormalProfilerBenchmarkCasesIT extends AbstractQwen3BenchmarkCasesIT {
    @Override
    protected String owner() {
        return "edwardcapriolo";
    }

    @Override
    protected String modelName() {
        return "Qwen3-4B-JQ4";
    }

    @Override
    protected Optional<String> configPath() {
        return Optional.of("benchmarks/configs/qwen3-4b-jq4.json");
    }
}
