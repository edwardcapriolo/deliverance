package io.teknek.deliverance.integration.qwen;

import org.junit.jupiter.api.Tag;

import java.util.Optional;

@Tag("large-model")
class NemotronArBenchmarkCasesIT extends AbstractQwen3BenchmarkCasesIT {
    @Override
    protected String owner() {
        return "nvidia";
    }

    @Override
    protected String modelName() {
        return "Nemotron-Labs-Diffusion-3B-Base-JQ4";
    }

    @Override
    protected Optional<String> configPath() {
        return Optional.of("benchmarks/configs/nemotron-labs-diffusion-3b-base-jq4-ar.json");
    }
}
