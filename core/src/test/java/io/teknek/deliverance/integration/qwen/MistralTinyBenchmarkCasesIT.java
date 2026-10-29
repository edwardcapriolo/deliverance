package io.teknek.deliverance.integration.qwen;

import org.junit.jupiter.api.Tag;

import java.util.Optional;

@Tag("small-model")
class MistralTinyBenchmarkCasesIT extends AbstractQwen3BenchmarkCasesIT {
    @Override
    protected String owner() {
        return "dacorvo";
    }

    @Override
    protected String modelName() {
        return "tiny-random-MistralForCausalLM";
    }

    @Override
    protected boolean downloadModel() {
        return true;
    }

    @Override
    protected Optional<String> expectedResponse(String caseId, int turn) {
        if ("builtin-reasoning-1".equals(caseId) && turn == 1) {
            return Optional.of("deltaPlvariable👍orfavoritesgaugesalesOrreplacingutfMiniavsp∆");
        }
        return Optional.empty();
    }
}
