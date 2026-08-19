package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.model.qwen3.Qwen3Config;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.io.File;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class QwenTensorParallelPreflightIT {
    @Test
    @Tag("longtest")
    void qwen06bJq4SupportsTwoFourAndEightTensorParallelRanks() throws Exception {
        File modelRoot = new ModelFetcher("edwardcapriolo", "Qwen3-0.6B-JQ4").maybeDownload();
        Qwen3Config config = JsonUtils.om.readValue(modelRoot.toPath().resolve("config.json").toFile(),
                Qwen3Config.class);

        for (int size : new int[]{2, 4, 8}) {
            assertTrue(TensorParallelPlanner.compatible(config, size), "Qwen3-0.6B-JQ4 should support tp=" + size);
        }
    }
}
