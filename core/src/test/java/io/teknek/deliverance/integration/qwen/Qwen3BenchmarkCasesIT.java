package io.teknek.deliverance.integration.qwen;

import org.junit.jupiter.api.Tag;

@Tag("large-model")
class Qwen3BenchmarkCasesIT extends AbstractQwen3BenchmarkCasesIT {
    @Override
    protected String owner() {
        return "edwardcapriolo";
    }

    @Override
    protected String modelName() {
        return "Qwen3-4B-JQ4";
    }

    @Override
    protected java.util.Optional<String> configPath() {
        return java.util.Optional.of("benchmarks/configs/qwen3-4b-jq4.json");
    }
}
