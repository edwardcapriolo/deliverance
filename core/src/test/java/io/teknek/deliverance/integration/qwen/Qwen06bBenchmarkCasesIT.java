package io.teknek.deliverance.integration.qwen;

import org.junit.jupiter.api.Tag;

@Tag("small-model")
class Qwen06bBenchmarkCasesIT extends AbstractQwen3BenchmarkCasesIT {
    @Override
    protected String owner() {
        return "edwardcapriolo";
    }

    @Override
    protected String modelName() {
        return "Qwen3-0.6B-JQ4";
    }
}
