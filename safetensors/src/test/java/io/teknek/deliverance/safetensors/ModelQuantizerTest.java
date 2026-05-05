package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ModelQuantizerTest {
    @TempDir
    Path tempDir;

    @Test
    public void keepsEmbeddingsDenseAndQuantizesMatrixWeights() throws Exception {
        Path sourceDir = tempDir.resolve("source");
        Path outputDir = tempDir.resolve("output");
        Files.createDirectories(sourceDir);
        Files.writeString(sourceDir.resolve("config.json"), "{\"model_type\":\"llama\"}");

        FloatBufferTensor embed = new FloatBufferTensor(1, 32);
        FloatBufferTensor proj = new FloatBufferTensor(1, 32);
        for (int i = 0; i < 32; i++) {
            embed.set(i, 0, i);
            proj.set(i - 8, 0, i);
        }
        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put("model.embed_tokens.weight", embed);
        tensors.put("model.layers.0.self_attn.q_proj.weight", proj);
        SafeTensorWriter.write(sourceDir.resolve("model.safetensors"), Map.of(), tensors);

        new ModelQuantizer(64).quantizeModelDirectory(sourceDir, outputDir);

        assertTrue(Files.exists(outputDir.resolve("config.json")));
        assertTrue(Files.exists(outputDir.resolve(SafeTensorIndexPojo.MODEL_INDEX_JSON)));
        try (DefaultWeightLoader loader = new DefaultWeightLoader(outputDir.toFile())) {
            assertEquals(DType.F32, loader.tensorInfoMap().get("model.embed_tokens.weight").dType);
            assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.self_attn.q_proj.weight").dType);
            assertTrue(loader.isWeightPresent("model.layers.0.self_attn.q_proj.weight.qb"));
        }
    }

    @Test
    public void quantizesUsingDeliveranceCachePaths() throws Exception {
        Path fakeHome = tempDir.resolve("home");
        String previousHome = System.getProperty("user.home");
        try {
            System.setProperty("user.home", fakeHome.toString());
            Path sourceDir = fakeHome.resolve(".deliverance").resolve("acme_demo-model");
            Path outputDir = fakeHome.resolve(".deliverance").resolve("acme_demo-model-q4");
            Files.createDirectories(sourceDir);
            Files.writeString(sourceDir.resolve("config.json"), "{\"model_type\":\"llama\"}");

            FloatBufferTensor proj = new FloatBufferTensor(1, 32);
            for (int i = 0; i < 32; i++) {
                proj.set(i - 8, 0, i);
            }
            Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
            tensors.put("model.layers.0.self_attn.q_proj.weight", proj);
            SafeTensorWriter.write(sourceDir.resolve("model.safetensors"), Map.of(), tensors);

            new ModelQuantizer(64).quantizeCachedModel("acme", "demo-model", "acme", "demo-model-q4");

            assertTrue(Files.exists(outputDir.resolve(SafeTensorIndexPojo.MODEL_INDEX_JSON)));
            try (DefaultWeightLoader loader = new DefaultWeightLoader(outputDir.toFile())) {
                assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.self_attn.q_proj.weight").dType);
            }
        } finally {
            if (previousHome == null) {
                System.clearProperty("user.home");
            } else {
                System.setProperty("user.home", previousHome);
            }
        }
    }
}
