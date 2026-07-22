package io.teknek.deliverance.safetensors;

import com.fasterxml.jackson.databind.JsonNode;
import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

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
        assertTrue(Files.exists(outputDir.resolve(".finished")));
        assertTrue(Files.exists(outputDir.resolve(SafeTensorIndexPojo.MODEL_INDEX_JSON)));
        try (DefaultWeightLoader loader = new DefaultWeightLoader(outputDir.toFile())) {
            assertEquals(DType.F32, loader.tensorInfoMap().get("model.embed_tokens.weight").dType);
            assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.self_attn.q_proj.weight").dType);
            assertTrue(loader.isWeightPresent("model.layers.0.self_attn.q_proj.weight.qb"));
        }
    }

    @Test
    public void writesQuantizationReadmeAndManifest() throws Exception {
        Path sourceDir = tempDir.resolve("source-manifest");
        Path outputDir = tempDir.resolve("output-manifest");
        Files.createDirectories(sourceDir);
        Files.writeString(sourceDir.resolve("config.json"), "{\"model_type\":\"llama\"}");
        Files.writeString(sourceDir.resolve("README.md"), "# Original Model\n\nModel card summary.\n\n## Original Details\n\nOriginal body.\n");

        FloatBufferTensor proj = new FloatBufferTensor(1, 32);
        for (int i = 0; i < 32; i++) {
            proj.set(i - 8, 0, i);
        }
        SafeTensorWriter.write(sourceDir.resolve("model.safetensors"), Map.of(),
                Map.of("model.layers.0.self_attn.q_proj.weight", proj));

        new ModelQuantizer(64).quantizeModelDirectory(sourceDir, outputDir);

        String readme = Files.readString(outputDir.resolve("README.md"));
        assertTrue(readme.startsWith("# Original Model"));
        assertTrue(readme.indexOf("# Deliverance Quantization") < readme.indexOf("## Original Details"));
        assertTrue(readme.contains("# Original Model"));
        assertTrue(readme.contains("## Original Details"));
        JsonNode manifest = JsonUtils.om.readTree(outputDir.resolve(ModelQuantizer.QUANTIZATION_MANIFEST).toFile());
        assertEquals(1, manifest.get("schemaVersion").asInt());
        assertEquals("Q4", manifest.get("targetType").asText());
        assertTrue(manifest.get("sourceSizeBytes").asLong() > 0);
        assertTrue(manifest.get("outputSizeBytes").asLong() > 0);
        JsonNode transform = manifest.get("tensorTransforms").get(0);
        assertEquals("model.layers.0.self_attn.q_proj.weight", transform.get("name").asText());
        assertEquals("F32", transform.get("sourceDType").asText());
        assertEquals("Q4", transform.get("outputDType").asText());
        assertTrue(transform.get("quantized").asBoolean());
        assertEquals("model.layers.0.self_attn.q_proj.weight.qb", transform.get("sidecars").get(0).asText());
    }

    @Test
    public void defaultFilterCoversDenseQwenMixtralAndQwenMoeProjectionWeights() {
        assertTrue(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.layers.0.self_attn.q_proj.weight"));
        assertTrue(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.layers.0.mlp.gate_proj.weight"));
        assertTrue(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.layers.19.mlp.experts.0.gate_proj.weight"));
        assertTrue(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.layers.19.mlp.experts.127.up_proj.weight"));
        assertTrue(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.layers.19.mlp.experts.127.down_proj.weight"));
        assertTrue(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.layers.3.block_sparse_moe.experts.0.w1.weight"));
        assertTrue(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.layers.3.block_sparse_moe.experts.7.w2.weight"));
        assertTrue(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.layers.3.block_sparse_moe.experts.7.w3.weight"));
        assertTrue(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.layers.3.shared_mlp.input_linear.weight"));
        assertTrue(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.layers.3.shared_mlp.output_linear.weight"));
        assertTrue(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.layers.3.block_sparse_moe.input_linear.weight"));
        assertTrue(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.layers.3.block_sparse_moe.output_linear.weight"));
        assertTrue(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.layers.3.mamba.in_proj.weight"));
        assertTrue(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.layers.3.mamba.out_proj.weight"));

        assertFalse(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.embed_tokens.weight"));
        assertFalse(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("lm_head.weight"));
        assertFalse(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.layers.0.input_layernorm.weight"));
        assertFalse(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.layers.19.mlp.gate.weight"));
        assertFalse(ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER.test("model.layers.3.block_sparse_moe.gate.weight"));
    }

    @Test
    public void quantizesQwenMoeExpertProjectionWeightsAndKeepsRouterDense() throws Exception {
        Path sourceDir = tempDir.resolve("source-qwen-moe");
        Path outputDir = tempDir.resolve("output-qwen-moe");
        Files.createDirectories(sourceDir);
        Files.writeString(sourceDir.resolve("config.json"), "{\"model_type\":\"qwen3_moe\"}");

        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put("model.layers.0.mlp.gate.weight", vector(1.0f));
        tensors.put("model.layers.0.mlp.experts.0.gate_proj.weight", vector(2.0f));
        tensors.put("model.layers.0.mlp.experts.0.up_proj.weight", vector(3.0f));
        tensors.put("model.layers.0.mlp.experts.0.down_proj.weight", vector(4.0f));
        SafeTensorWriter.write(sourceDir.resolve("model.safetensors"), Map.of(), tensors);

        new ModelQuantizer(256).quantizeModelDirectory(sourceDir, outputDir);

        try (DefaultWeightLoader loader = new DefaultWeightLoader(outputDir.toFile())) {
            assertEquals(DType.F32, loader.tensorInfoMap().get("model.layers.0.mlp.gate.weight").dType);
            assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.mlp.experts.0.gate_proj.weight").dType);
            assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.mlp.experts.0.up_proj.weight").dType);
            assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.mlp.experts.0.down_proj.weight").dType);
            assertTrue(loader.isWeightPresent("model.layers.0.mlp.experts.0.gate_proj.weight.qb"));
            assertTrue(loader.isWeightPresent("model.layers.0.mlp.experts.0.up_proj.weight.qb"));
            assertTrue(loader.isWeightPresent("model.layers.0.mlp.experts.0.down_proj.weight.qb"));
        }

        JsonNode manifest = JsonUtils.om.readTree(outputDir.resolve(ModelQuantizer.QUANTIZATION_MANIFEST).toFile());
        long quantizedCount = 0;
        for (JsonNode transform : manifest.get("tensorTransforms")) {
            if (transform.get("quantized").asBoolean()) {
                quantizedCount++;
            }
        }
        assertEquals(3, quantizedCount);
    }

    @Test
    public void quantizesMixtralExpertProjectionWeightsAndKeepsRouterDense() throws Exception {
        Path sourceDir = tempDir.resolve("source-mixtral-moe");
        Path outputDir = tempDir.resolve("output-mixtral-moe");
        Files.createDirectories(sourceDir);
        Files.writeString(sourceDir.resolve("config.json"), "{\"model_type\":\"mixtral\"}");

        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put("model.layers.0.block_sparse_moe.gate.weight", vector(1.0f));
        tensors.put("model.layers.0.block_sparse_moe.experts.0.w1.weight", vector(2.0f));
        tensors.put("model.layers.0.block_sparse_moe.experts.0.w2.weight", vector(3.0f));
        tensors.put("model.layers.0.block_sparse_moe.experts.0.w3.weight", vector(4.0f));
        SafeTensorWriter.write(sourceDir.resolve("model.safetensors"), Map.of(), tensors);

        new ModelQuantizer(256).quantizeModelDirectory(sourceDir, outputDir);

        try (DefaultWeightLoader loader = new DefaultWeightLoader(outputDir.toFile())) {
            assertEquals(DType.F32, loader.tensorInfoMap().get("model.layers.0.block_sparse_moe.gate.weight").dType);
            assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.block_sparse_moe.experts.0.w1.weight").dType);
            assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.block_sparse_moe.experts.0.w2.weight").dType);
            assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.block_sparse_moe.experts.0.w3.weight").dType);
        }
    }

    @Test
    public void quantizesGraniteMoeHybridProjectionWeights() throws Exception {
        Path sourceDir = tempDir.resolve("source-granite");
        Path outputDir = tempDir.resolve("output-granite");
        Files.createDirectories(sourceDir);
        Files.writeString(sourceDir.resolve("config.json"), "{\"model_type\":\"granitemoehybrid\"}");

        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put("model.layers.0.shared_mlp.input_linear.weight", vector(1.0f));
        tensors.put("model.layers.0.shared_mlp.output_linear.weight", vector(2.0f));
        tensors.put("model.layers.0.mamba.in_proj.weight", vector(3.0f));
        tensors.put("model.layers.0.mamba.out_proj.weight", vector(4.0f));
        tensors.put("model.layers.0.mamba.conv1d.weight", conv3d());
        SafeTensorWriter.write(sourceDir.resolve("model.safetensors"), Map.of(), tensors);

        new ModelQuantizer(256).quantizeModelDirectory(sourceDir, outputDir);

        try (DefaultWeightLoader loader = new DefaultWeightLoader(outputDir.toFile())) {
            assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.shared_mlp.input_linear.weight").dType);
            assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.shared_mlp.output_linear.weight").dType);
            assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.mamba.in_proj.weight").dType);
            assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.mamba.out_proj.weight").dType);
            assertEquals(DType.F32, loader.tensorInfoMap().get("model.layers.0.mamba.conv1d.weight").dType);
            assertTrue(loader.isWeightPresent("model.layers.0.shared_mlp.input_linear.weight.qb"));
        }
    }

    @Test
    public void canUseDefaultWeightLoaderReadMode() throws Exception {
        Path sourceDir = tempDir.resolve("source-default-mode");
        Path outputDir = tempDir.resolve("output-default-mode");
        Files.createDirectories(sourceDir);
        Files.writeString(sourceDir.resolve("config.json"), "{\"model_type\":\"llama\"}");

        SafeTensorWriter.write(sourceDir.resolve("model.safetensors"), Map.of(),
                Map.of("model.layers.0.self_attn.q_proj.weight", vector(1.0f)));

        new ModelQuantizer(256, ModelQuantizer.ReadMode.DEFAULT_WEIGHT_LOADER).quantizeModelDirectory(sourceDir, outputDir);

        try (DefaultWeightLoader loader = new DefaultWeightLoader(outputDir.toFile())) {
            assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.self_attn.q_proj.weight").dType);
        }
    }

    @Test
    public void shardReadModeProcessesMultiShardModel() throws Exception {
        Path sourceDir = tempDir.resolve("source-shard-mode");
        Path outputDir = tempDir.resolve("output-shard-mode");
        Files.createDirectories(sourceDir);
        Files.writeString(sourceDir.resolve("config.json"), "{\"model_type\":\"qwen3_moe\"}");

        SafeTensorWriter.writeShardFile(sourceDir.resolve("model-00001-of-00002.safetensors"), Map.of(),
                SafeTensorWriter.flatten(Map.of("model.layers.0.self_attn.q_proj.weight", vector(1.0f))));
        SafeTensorWriter.writeShardFile(sourceDir.resolve("model-00002-of-00002.safetensors"), Map.of(),
                SafeTensorWriter.flatten(Map.of("model.layers.0.mlp.experts.0.gate_proj.weight", vector(2.0f))));
        JsonUtils.om.writeValue(sourceDir.resolve(SafeTensorIndexPojo.MODEL_INDEX_JSON).toFile(),
                new SafeTensorIndexPojo(Map.of(), Map.of(
                        "model.layers.0.self_attn.q_proj.weight", "model-00001-of-00002.safetensors",
                        "model.layers.0.mlp.experts.0.gate_proj.weight", "model-00002-of-00002.safetensors")));

        new ModelQuantizer(256, ModelQuantizer.ReadMode.SHARD_WEIGHT_LOADER).quantizeModelDirectory(sourceDir, outputDir);

        try (DefaultWeightLoader loader = new DefaultWeightLoader(outputDir.toFile())) {
            assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.self_attn.q_proj.weight").dType);
            assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.mlp.experts.0.gate_proj.weight").dType);
            assertTrue(loader.isWeightPresent("model.layers.0.self_attn.q_proj.weight.qb"));
            assertTrue(loader.isWeightPresent("model.layers.0.mlp.experts.0.gate_proj.weight.qb"));
        }
    }

    @Test
    public void keepsNormWeightsDenseEvenIfStoredAsRowVectors() throws Exception {
        Path sourceDir = tempDir.resolve("source-norm");
        Path outputDir = tempDir.resolve("output-norm");
        Files.createDirectories(sourceDir);
        Files.writeString(sourceDir.resolve("config.json"), "{\"model_type\":\"llama\"}");

        FloatBufferTensor norm = new FloatBufferTensor(1, 32);
        FloatBufferTensor proj = new FloatBufferTensor(1, 32);
        for (int i = 0; i < 32; i++) {
            norm.set(1.0f, 0, i);
            proj.set(i - 8, 0, i);
        }
        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put("model.layers.0.input_layernorm.weight", norm);
        tensors.put("model.layers.0.self_attn.q_proj.weight", proj);
        SafeTensorWriter.write(sourceDir.resolve("model.safetensors"), Map.of(), tensors);

        new ModelQuantizer(64).quantizeModelDirectory(sourceDir, outputDir);

        try (DefaultWeightLoader loader = new DefaultWeightLoader(outputDir.toFile())) {
            assertEquals(DType.F32, loader.tensorInfoMap().get("model.layers.0.input_layernorm.weight").dType);
            assertEquals(DType.Q4, loader.tensorInfoMap().get("model.layers.0.self_attn.q_proj.weight").dType);
        }
    }

    @Test
    public void keepsNonAllowlistedTwoDimensionalWeightsDense() throws Exception {
        Path sourceDir = tempDir.resolve("source-nonallow");
        Path outputDir = tempDir.resolve("output-nonallow");
        Files.createDirectories(sourceDir);
        Files.writeString(sourceDir.resolve("config.json"), "{\"model_type\":\"llama\"}");

        FloatBufferTensor misc = new FloatBufferTensor(1, 32);
        for (int i = 0; i < 32; i++) {
            misc.set(i + 1, 0, i);
        }
        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put("model.layers.0.some_other.weight", misc);
        SafeTensorWriter.write(sourceDir.resolve("model.safetensors"), Map.of(), tensors);

        new ModelQuantizer(64).quantizeModelDirectory(sourceDir, outputDir);

        try (DefaultWeightLoader loader = new DefaultWeightLoader(outputDir.toFile())) {
            assertEquals(DType.F32, loader.tensorInfoMap().get("model.layers.0.some_other.weight").dType);
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

            new ModelQuantizer(16).quantizeCachedModel("acme", "demo-model", "acme", "demo-model-q4");

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

    @Test
    public void keepsNonTwoDimensionalWeightsDense() throws Exception {
        Path sourceDir = tempDir.resolve("source-3d");
        Path outputDir = tempDir.resolve("output-3d");
        Files.createDirectories(sourceDir);
        Files.writeString(sourceDir.resolve("config.json"), "{\"model_type\":\"llama\"}");

        FloatBufferTensor conv = new FloatBufferTensor(2, 2, 8);
        for (int i = 0; i < 32; i++) {
            conv.set(i - 4, 0, i / 8, i % 8);
        }
        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put("model.audio_tower.layers.8.lconv1d.depthwise_conv1d.weight", conv);
        SafeTensorWriter.write(sourceDir.resolve("model.safetensors"), Map.of(), tensors);

        new ModelQuantizer(64).quantizeModelDirectory(sourceDir, outputDir);

        try (DefaultWeightLoader loader = new DefaultWeightLoader(outputDir.toFile())) {
            assertEquals(DType.F32, loader.tensorInfoMap().get("model.audio_tower.layers.8.lconv1d.depthwise_conv1d.weight").dType);
        }
    }

    @Test
    public void skipsLogicalParentNameForSplitTensors() throws Exception {
        Path sourceDir = tempDir.resolve("source-split");
        Files.createDirectories(sourceDir);
        Files.writeString(sourceDir.resolve("config.json"), "{\"model_type\":\"llama\"}");

        FloatBufferTensor part0 = new FloatBufferTensor(2, 4);
        FloatBufferTensor part1 = new FloatBufferTensor(3, 4);
        SafeTensorWriter.writeShardFile(
                sourceDir.resolve("model-00001-of-00002.safetensors"),
                Map.of(),
                SafeTensorWriter.flatten(Map.of("logical.weight-part-0", part0))
        );
        SafeTensorWriter.writeShardFile(
                sourceDir.resolve("model-00002-of-00002.safetensors"),
                Map.of(),
                SafeTensorWriter.flatten(Map.of("logical.weight-part-1", part1))
        );
        JsonUtils.om.writeValue(sourceDir.resolve(SafeTensorIndexPojo.MODEL_INDEX_JSON).toFile(),
                new SafeTensorIndexPojo(
                        Map.of(),
                        Map.of(
                                "logical.weight-part-0", "model-00001-of-00002.safetensors",
                                "logical.weight-part-1", "model-00002-of-00002.safetensors"
                        )
                ));

        try (DefaultWeightLoader loader = new DefaultWeightLoader(sourceDir.toFile())) {
            tensorInfoMap(loader).put("logical.weight", new io.teknek.deliverance.tensor.TensorInfo(DType.F32, new long[]{5, 4}, new long[]{0, 80}));
            ModelQuantizer quantizer = new ModelQuantizer();
            assertTrue(quantizer.isLogicalSplitTensor("logical.weight", loader));
            assertFalse(quantizer.isLogicalSplitTensor("logical.weight-part-0", loader));
        }
    }

    @Disabled
    //@Test
    void fullModelQuantizerTest(){
        //google_gemma-4-E2B-it
        new ModelQuantizer().quantizeCachedModel("google", "gemma-4-E2B-it",
                "edward", "gemma-4-E2B-it-JQ4", DType.Q4,
                ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER);
    }

    @Disabled
    void fullModelQuantizerTest2() {
        //google_gemma-4-E2B-it
        new ModelQuantizer().quantizeCachedModel("google", "gemma-4-E4B-it",
                "edward", "gemma-4-E4B-it-JQ4", DType.Q4,
                ModelQuantizer.DEFAULT_Q4_TENSOR_FILTER);
    }


    @SuppressWarnings("unchecked")
    private static Map<String, io.teknek.deliverance.tensor.TensorInfo> tensorInfoMap(DefaultWeightLoader loader) throws Exception {
        Field field = DefaultWeightLoader.class.getDeclaredField("allTensorInfoMap");
        field.setAccessible(true);
        return (Map<String, io.teknek.deliverance.tensor.TensorInfo>) field.get(loader);
    }

    private static FloatBufferTensor vector(float offset) {
        FloatBufferTensor tensor = new FloatBufferTensor(1, 32);
        for (int i = 0; i < 32; i++) {
            tensor.set(offset + i, 0, i);
        }
        return tensor;
    }

    private static FloatBufferTensor conv3d() {
        FloatBufferTensor tensor = new FloatBufferTensor(2, 2, 8);
        for (int i = 0; i < 32; i++) {
            tensor.set(i, 0, i / 8, i % 8);
        }
        return tensor;
    }
}
