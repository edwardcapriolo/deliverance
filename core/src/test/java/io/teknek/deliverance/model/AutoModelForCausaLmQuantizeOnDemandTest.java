package io.teknek.deliverance.model;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.safetensors.ModelQuantizer;
import io.teknek.deliverance.safetensors.SafeTensorWriter;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AutoModelForCausaLmQuantizeOnDemandTest {
    @TempDir
    Path tempDir;

    @Test
    public void usesExistingQuantizedTargetWithoutFetchingSource() throws Exception {
        Path cacheDir = tempDir.resolve("cache");
        Path targetDir = cacheDir.resolve("acme_demo-q4");
        Files.createDirectories(targetDir);
        Files.writeString(targetDir.resolve(".finished"), "");

        ModelFetcher source = new ModelFetcher("acme", "missing-source");
        source.setBaseDir(cacheDir);

        ModelFetcher resolved = AutoModelForCausaLm.newBuilder(source)
                .withDownload(false)
                .withQuantizeOnDemand(DType.Q4, "acme", "demo-q4")
                .resolveModelFetcherForLoad();

        assertEquals(targetDir, resolved.pathForModel());
        assertFalse(Files.exists(cacheDir.resolve("acme_missing-source")));
    }

    @Test
    public void quantizesMissingTargetFromLocalSourceWhenDownloadsDisabled() throws Exception {
        Path cacheDir = tempDir.resolve("cache");
        Path sourceDir = cacheDir.resolve("acme_demo");
        Path targetDir = cacheDir.resolve("acme_demo-q4");
        createTinyModel(sourceDir);

        ModelFetcher source = new ModelFetcher("acme", "demo");
        source.setBaseDir(cacheDir);

        ModelFetcher resolved = AutoModelForCausaLm.newBuilder(source)
                .withDownload(false)
                .withQuantizeOnDemand(DType.Q4, "acme", "demo-q4")
                .resolveModelFetcherForLoad();

        assertEquals(targetDir, resolved.pathForModel());
        assertTrue(Files.exists(targetDir.resolve(ModelQuantizer.QUANTIZATION_MANIFEST)));
        assertTrue(Files.exists(targetDir.resolve(".finished")));
        try (DefaultWeightLoader loader = new DefaultWeightLoader(targetDir.toFile())) {
            assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.self_attn.q_proj.weight").dType);
            assertTrue(loader.isWeightPresent("model.layers.0.self_attn.q_proj.weight.qb"));
        }
    }

    private static void createTinyModel(Path sourceDir) throws Exception {
        Files.createDirectories(sourceDir);
        Files.writeString(sourceDir.resolve("config.json"), "{\"model_type\":\"llama\"}");
        Files.writeString(sourceDir.resolve("tokenizer.json"), "{}");
        FloatBufferTensor proj = new FloatBufferTensor(1, 32);
        for (int i = 0; i < 32; i++) {
            proj.set(i - 8, 0, i);
        }
        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put("model.layers.0.self_attn.q_proj.weight", proj);
        SafeTensorWriter.write(sourceDir.resolve("model.safetensors"), Map.of(), tensors);
    }
}
