package io.teknek.deliverance.model;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

public class AutoModelConfigTest {
    @TempDir
    Path tempDir;

    @Test
    public void readsJsonAndAppliesBuilderSettings() throws Exception {
        Path configFile = tempDir.resolve("auto-model.json");
        Files.writeString(configFile, """
                {
                  "workingMemoryType": "BF16",
                  "workingQuantType": "Q4",
                  "outputHeadQuantization": "Q4",
                  "download": false,
                  "maxBatchSize": 17,
                  "kvBufferCache": {
                    "maxEntries": 0,
                    "blockSize": 16,
                    "maxPrefixTokensPerPrompt": 128,
                    "prefixCheckpointPolicy": "FIXED_BLOCKS",
                    "maxPrefixCheckpointsPerPrompt": 5,
                    "prefixCheckpointAnchors": [16, 32, 64],
                    "contextRowsPerPageTarget": 64
                  }
                }
                """);

        AutoModelConfig config = AutoModelConfig.fromJson(configFile);
        AutoModelForCausaLm.Builder builder = AutoModelForCausaLm.newBuilder(new ModelFetcher("Qwen", "Qwen3-0.6B"))
                .withConfig(config);

        assertEquals(DType.BF16, builder.getWorkingMem());
        assertEquals(DType.Q4, builder.getWorkingQuant());
        assertEquals(DType.Q4, builder.getOutputHeadQuantization().orElseThrow());
        assertFalse(builder.isDownload());
        assertEquals(17, builder.getMaxBatchSize());
        assertEquals(0, builder.getSettings().getMaxEntries());
        assertEquals(16, builder.getSettings().getBlockSize());
        assertEquals(128, builder.getSettings().getMaxPrefixTokensPerPrompt());
        assertEquals(io.teknek.deliverance.tensor.KvBufferCacheSettings.PrefixCheckpointPolicy.FIXED_BLOCKS,
                builder.getSettings().getPrefixCheckpointPolicy());
        assertEquals(5, builder.getSettings().getMaxPrefixCheckpointsPerPrompt());
        assertEquals(java.util.List.of(16, 32, 64), builder.getSettings().getPrefixCheckpointAnchors());
        assertEquals(64, builder.getSettings().getContextRowsPerPageTarget());
    }
}
