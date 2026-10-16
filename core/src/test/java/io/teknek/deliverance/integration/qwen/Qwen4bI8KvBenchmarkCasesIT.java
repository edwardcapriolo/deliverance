package io.teknek.deliverance.integration.qwen;

import org.junit.jupiter.api.Tag;

import java.util.Optional;

/** Runs the standard Qwen benchmark cases with I8 key/value cache storage enabled. */
@Tag("large-model")
class Qwen4bI8KvBenchmarkCasesIT extends AbstractQwen3BenchmarkCasesIT {
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
        return Optional.of("benchmarks/configs/qwen3-4b-jq4-i8-kv.json");
    }
}
