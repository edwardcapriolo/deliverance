package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link ResolvedLoraAdapter} (Phase 2 runtime hot-swap), no network access --
 * see step 4 plan Section 8. Uses the same synthetic-adapter-directory pattern as {@link
 * LoraAdapterTest}.
 */
public class ResolvedLoraAdapterTest {

    @TempDir
    Path tempDir;

    private static final String Q_PROJ = "model.layers.0.self_attn.q_proj.weight";
    private static final String K_PROJ = "model.layers.0.self_attn.k_proj.weight";

    @Test
    void deltaForCachesTheResolvedDeltaAcrossCalls() throws IOException {
        writeAdapterConfig(tempDir, 4, 8.0, "q_proj");
        writeQProjTensors(tempDir, 4, 8, 6);

        try (LoraAdapter adapter = LoraAdapter.load(tempDir.toFile());
                ResolvedLoraAdapter resolved = new ResolvedLoraAdapter(adapter, DType.F32)) {
            Optional<LoraLayerDelta> first = resolved.deltaFor(Q_PROJ);
            Optional<LoraLayerDelta> second = resolved.deltaFor(Q_PROJ);

            assertTrue(first.isPresent());
            assertSame(first.get(), second.get(), "second call must return the cached instance, not re-resolve");
        }
    }

    @Test
    void scaledLoraBIsPreScaledByAdapterScale() throws IOException {
        int rank = 4;
        double alpha = 8.0; // scale = alpha / r = 2.0
        writeAdapterConfig(tempDir, rank, alpha, "q_proj");

        int inFeatures = 8;
        int outFeatures = 6;
        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        FloatBufferTensor loraB = new FloatBufferTensor(outFeatures, rank);
        for (int row = 0; row < outFeatures; row++) {
            for (int col = 0; col < rank; col++) {
                loraB.set((row * rank + col) + 1.0f, row, col);
            }
        }
        tensors.put(LoraTensorNames.loraA(Q_PROJ), new FloatBufferTensor(rank, inFeatures));
        tensors.put(LoraTensorNames.loraB(Q_PROJ), loraB);
        SafeTensorWriter.write(tempDir.resolve(LoraAdapter.SAFETENSORS_FILE_NAME), Map.of(), tensors);

        try (LoraAdapter adapter = LoraAdapter.load(tempDir.toFile());
                ResolvedLoraAdapter resolved = new ResolvedLoraAdapter(adapter, DType.F32)) {
            LoraLayerDelta delta = resolved.deltaFor(Q_PROJ).orElseThrow();
            assertEquals(rank, delta.rank());
            for (int row = 0; row < outFeatures; row++) {
                for (int col = 0; col < rank; col++) {
                    float expected = ((row * rank + col) + 1.0f) * 2.0f;
                    assertEquals(expected, delta.scaledLoraB().get(row, col), 1e-6f);
                }
            }
        }
    }

    @Test
    void nonTargetedNameCachesEmpty() throws IOException {
        writeAdapterConfig(tempDir, 4, 8.0, "q_proj");
        writeQProjTensors(tempDir, 4, 8, 6);

        try (LoraAdapter adapter = LoraAdapter.load(tempDir.toFile());
                ResolvedLoraAdapter resolved = new ResolvedLoraAdapter(adapter, DType.F32)) {
            assertFalse(resolved.deltaFor(K_PROJ).isPresent());
            assertFalse(resolved.deltaFor(K_PROJ).isPresent());
        }
    }

    private static void writeQProjTensors(Path adapterDir, int rank, int inFeatures, int outFeatures) throws IOException {
        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put(LoraTensorNames.loraA(Q_PROJ), new FloatBufferTensor(rank, inFeatures));
        tensors.put(LoraTensorNames.loraB(Q_PROJ), new FloatBufferTensor(outFeatures, rank));
        SafeTensorWriter.write(adapterDir.resolve(LoraAdapter.SAFETENSORS_FILE_NAME), Map.of(), tensors);
    }

    private static void writeAdapterConfig(Path adapterDir, int rank, double alpha, String... targetModules)
            throws IOException {
        StringBuilder modules = new StringBuilder();
        for (int i = 0; i < targetModules.length; i++) {
            if (i > 0) modules.append(",");
            modules.append("\"").append(targetModules[i]).append("\"");
        }
        String json = "{\"r\": " + rank + ", \"lora_alpha\": " + alpha + ", \"target_modules\": [" + modules + "]}";
        Files.writeString(adapterDir.resolve(LoraAdapterConfig.FILE_NAME), json);
    }
}
