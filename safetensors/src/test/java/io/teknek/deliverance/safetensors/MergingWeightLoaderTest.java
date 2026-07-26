package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.TensorInfo;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.tensor.operations.NaiveTensorOperations;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Verifies {@link MergingWeightLoader}'s merge math against a naive plain-Java reference
 * calculation on small synthetic tensors, per {@code
 * StepPlans/deliverance_lora_step3_merging_weightloader_plan_v1.md} Section 7. No network access;
 * base model and adapter are both synthetic directories built with {@link SafeTensorWriter}, the
 * same pattern {@link LoraAdapterTest} already uses for its no-network cases.
 */
public class MergingWeightLoaderTest {

    private static final String Q_PROJ = "model.layers.0.self_attn.q_proj.weight";
    private static final String K_PROJ = "model.layers.0.self_attn.k_proj.weight";
    private static final int OUT_FEATURES = 4;
    private static final int IN_FEATURES = 4;
    private static final int RANK = 2;

    @TempDir
    Path tempDir;

    @Test
    void mergesDeltaIntoTargetedModule() throws IOException {
        Path baseDir = tempDir.resolve("base");
        Path adapterDir = tempDir.resolve("adapter");

        float[][] base = matrix(new float[][] {
                { 1f, 2f, 3f, 4f },
                { 5f, 6f, 7f, 8f },
                { 9f, 10f, 11f, 12f },
                { 13f, 14f, 15f, 16f } });
        float[][] loraA = matrix(new float[][] {
                { 0.1f, 0.2f, 0.3f, 0.4f },
                { 0.5f, 0.6f, 0.7f, 0.8f } });
        float[][] loraB = matrix(new float[][] {
                { 1f, 2f },
                { 3f, 4f },
                { 5f, 6f },
                { 7f, 8f } });
        double alpha = 4.0;
        double rank = RANK;
        double scale = alpha / rank;

        writeBaseModel(baseDir, Map.of(Q_PROJ, base, K_PROJ, base));
        writeAdapter(adapterDir, RANK, alpha, "q_proj", Map.of(
                LoraTensorNames.loraA(Q_PROJ), loraA,
                LoraTensorNames.loraB(Q_PROJ), loraB));

        float[][] expected = new float[OUT_FEATURES][IN_FEATURES];
        for (int o = 0; o < OUT_FEATURES; o++) {
            for (int i = 0; i < IN_FEATURES; i++) {
                double delta = 0;
                for (int k = 0; k < RANK; k++) {
                    delta += loraB[o][k] * loraA[k][i];
                }
                expected[o][i] = (float) (base[o][i] + scale * delta);
            }
        }

        try (DefaultWeightLoader delegate = DefaultWeightLoader.open(baseDir.toFile());
                LoraAdapter adapter = LoraAdapter.load(adapterDir.toFile())) {
            MergingWeightLoader merging = new MergingWeightLoader(delegate, adapter, new NaiveTensorOperations());
            try (AbstractTensor merged = merging.load(Q_PROJ)) {
                for (int o = 0; o < OUT_FEATURES; o++) {
                    for (int i = 0; i < IN_FEATURES; i++) {
                        assertEquals(expected[o][i], merged.get(o, i), 1e-4f,
                                "mismatch at [" + o + "," + i + "]");
                    }
                }
            }
        }
    }

    @Test
    void passesThroughNonTargetedModuleUnchanged() throws IOException {
        Path baseDir = tempDir.resolve("base");
        Path adapterDir = tempDir.resolve("adapter");

        float[][] base = matrix(new float[][] {
                { 1f, 2f, 3f, 4f },
                { 5f, 6f, 7f, 8f },
                { 9f, 10f, 11f, 12f },
                { 13f, 14f, 15f, 16f } });
        float[][] loraA = matrix(new float[][] { { 0.1f, 0.2f, 0.3f, 0.4f }, { 0.5f, 0.6f, 0.7f, 0.8f } });
        float[][] loraB = matrix(new float[][] { { 1f, 2f }, { 3f, 4f }, { 5f, 6f }, { 7f, 8f } });

        writeBaseModel(baseDir, Map.of(Q_PROJ, base, K_PROJ, base));
        writeAdapter(adapterDir, RANK, 4.0, "q_proj", Map.of(
                LoraTensorNames.loraA(Q_PROJ), loraA,
                LoraTensorNames.loraB(Q_PROJ), loraB));

        try (DefaultWeightLoader delegate = DefaultWeightLoader.open(baseDir.toFile());
                LoraAdapter adapter = LoraAdapter.load(adapterDir.toFile())) {
            MergingWeightLoader merging = new MergingWeightLoader(delegate, adapter, new NaiveTensorOperations());
            try (AbstractTensor unchanged = merging.load(K_PROJ)) {
                for (int o = 0; o < OUT_FEATURES; o++) {
                    for (int i = 0; i < IN_FEATURES; i++) {
                        assertEquals(base[o][i], unchanged.get(o, i), 1e-6f);
                    }
                }
            }
        }
    }

    @Test
    void rejectsPreQuantizedOnDiskBaseModel() {
        WeightLoader q4Delegate = new WeightLoader() {
            @Override
            public Map<String, String> metadata() {
                return Map.of();
            }

            @Override
            public Map<String, TensorInfo> tensorInfoMap() {
                return Map.of();
            }

            @Override
            public DType getModelDType() {
                return DType.Q4;
            }

            @Override
            public void close() {
            }
        };
        LoraAdapter adapter = null;
        try {
            Path adapterDir = tempDir.resolve("adapter");
            float[][] loraA = matrix(new float[][] { { 0.1f, 0.2f, 0.3f, 0.4f }, { 0.5f, 0.6f, 0.7f, 0.8f } });
            float[][] loraB = matrix(new float[][] { { 1f, 2f }, { 3f, 4f }, { 5f, 6f }, { 7f, 8f } });
            writeAdapter(adapterDir, RANK, 4.0, "q_proj", Map.of(
                    LoraTensorNames.loraA(Q_PROJ), loraA,
                    LoraTensorNames.loraB(Q_PROJ), loraB));
            adapter = LoraAdapter.load(adapterDir.toFile());
            LoraAdapter finalAdapter = adapter;
            assertThrows(UnsupportedOperationException.class,
                    () -> new MergingWeightLoader(q4Delegate, finalAdapter, new NaiveTensorOperations()));
        } catch (IOException e) {
            throw new RuntimeException(e);
        } finally {
            if (adapter != null) {
                adapter.close();
            }
        }
    }

    private static float[][] matrix(float[][] values) {
        return values;
    }

    private static void writeBaseModel(Path baseDir, Map<String, float[][]> tensors) {
        Map<String, AbstractTensor> asTensors = new LinkedHashMap<>();
        for (Map.Entry<String, float[][]> entry : tensors.entrySet()) {
            asTensors.put(entry.getKey(), toTensor(entry.getValue()));
        }
        SafeTensorWriter.writeModel(baseDir, Map.of(), asTensors);
    }

    private static void writeAdapter(Path adapterDir, int rank, double alpha, String targetModule,
            Map<String, float[][]> tensors) throws IOException {
        Files.createDirectories(adapterDir);
        String json = "{\"r\": " + rank + ", \"lora_alpha\": " + alpha + ", \"target_modules\": [\"" + targetModule + "\"]}";
        Files.writeString(adapterDir.resolve(LoraAdapterConfig.FILE_NAME), json);

        Map<String, AbstractTensor> asTensors = new LinkedHashMap<>();
        for (Map.Entry<String, float[][]> entry : tensors.entrySet()) {
            asTensors.put(entry.getKey(), toTensor(entry.getValue()));
        }
        SafeTensorWriter.write(adapterDir.resolve(LoraAdapter.SAFETENSORS_FILE_NAME), Map.of(), asTensors);
    }

    private static AbstractTensor toTensor(float[][] values) {
        FloatBufferTensor tensor = new FloatBufferTensor(values.length, values[0].length);
        for (int r = 0; r < values.length; r++) {
            for (int c = 0; c < values[r].length; c++) {
                tensor.set(values[r][c], r, c);
            }
        }
        return tensor;
    }
}
