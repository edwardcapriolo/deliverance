package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.impl.FloatBufferTensor;
import io.teknek.deliverance.safetensors.fetch.LoraAdapterModelFetcher;
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
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LoraAdapterTest {

    @TempDir
    Path tempDir;

    /**
     * No network access -- builds a tiny synthetic adapter directory with {@link
     * SafeTensorWriter} the same way other tests in this module construct synthetic
     * safetensors fixtures (see {@code SafeTensorWriterTest}), so this test runs fast and
     * reliably in CI. See {@link #fromPretrainedDownloadsAndParsesARealPublishedAdapter()}
     * for the real-HF-adapter counterpart.
     */
    @Test
    void loadParsesASyntheticSingleModuleAdapter() throws IOException {
        writeAdapterConfig(tempDir, 4, 8.0, "q_proj");

        String base = "model.layers.0.self_attn.q_proj.weight";
        int rankDim = 4;
        int inFeatures = 8;
        int outFeatures = 6;
        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put(LoraTensorNames.loraA(base), new FloatBufferTensor(rankDim, inFeatures));
        tensors.put(LoraTensorNames.loraB(base), new FloatBufferTensor(outFeatures, rankDim));
        SafeTensorWriter.write(tempDir.resolve(LoraAdapter.SAFETENSORS_FILE_NAME), Map.of(), tensors);

        try (LoraAdapter adapter = LoraAdapter.load(tempDir.toFile())) {
            assertEquals(4, adapter.rank());
            assertEquals(8.0, adapter.alpha());
            assertEquals(2.0, adapter.scale());

            Optional<LoraAdapter.LoraDelta> delta = adapter.deltaFor(base);
            assertTrue(delta.isPresent());
            assertEquals(rankDim, delta.get().loraA().shape().first());
            assertEquals(inFeatures, delta.get().loraA().shape().last());
            assertEquals(outFeatures, delta.get().loraB().shape().first());
            assertEquals(rankDim, delta.get().loraB().shape().last());

            // k_proj was never a target_modules entry for this adapter.
            assertFalse(adapter.deltaFor("model.layers.0.self_attn.k_proj.weight").isPresent());
        }
    }

    @Test
    void loadFailsFastWhenTargetModuleHasNoMatchingTensors() throws IOException {
        // Config claims to target k_proj, but the safetensors file only has a q_proj pair --
        // this must fail loudly at construction time, not lazily on first deltaFor() call.
        writeAdapterConfig(tempDir, 4, 8.0, "k_proj");

        String base = "model.layers.0.self_attn.q_proj.weight";
        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put(LoraTensorNames.loraA(base), new FloatBufferTensor(4, 8));
        tensors.put(LoraTensorNames.loraB(base), new FloatBufferTensor(6, 4));
        SafeTensorWriter.write(tempDir.resolve(LoraAdapter.SAFETENSORS_FILE_NAME), Map.of(), tensors);

        assertThrows(IllegalStateException.class, () -> LoraAdapter.load(tempDir.toFile()));
    }

    @Test
    void loadFailsFastOnRankMismatchBetweenConfigAndTensors() throws IOException {
        // Config declares rank 4, but the tensors on disk are actually rank 2.
        writeAdapterConfig(tempDir, 4, 8.0, "q_proj");

        String base = "model.layers.0.self_attn.q_proj.weight";
        Map<String, AbstractTensor> tensors = new LinkedHashMap<>();
        tensors.put(LoraTensorNames.loraA(base), new FloatBufferTensor(2, 8));
        tensors.put(LoraTensorNames.loraB(base), new FloatBufferTensor(6, 2));
        SafeTensorWriter.write(tempDir.resolve(LoraAdapter.SAFETENSORS_FILE_NAME), Map.of(), tensors);

        assertThrows(IllegalStateException.class, () -> LoraAdapter.load(tempDir.toFile()));
    }

    /**
     * Integration test against a real, small, publicly published PEFT LoRA adapter for
     * Llama-3.2-1B (rank 16, ~45MB, no {@code modules_to_save} bloat -- verified before use,
     * see the parent plan and {@code LoraTensorNamesTest}'s naming-convention test, both
     * checked against this adapter's actual safetensors header). Requires network access.
     */
    @Test
    void fromPretrainedDownloadsAndParsesARealPublishedAdapter() {
        // Uses the default ~/.deliverance cache (like the project's other HF-backed tests,
        // e.g. LeafModelExample) rather than an isolated temp dir, so repeated local/CI runs
        // reuse the download instead of re-fetching ~45MB every time.
        LoraAdapterModelFetcher fetcher = new LoraAdapterModelFetcher("bunnycore", "Llama-3.2-1b-chatml-lora_model");

        try (LoraAdapter adapter = LoraAdapter.fromPretrained(fetcher)) {
            assertEquals(16, adapter.rank());
            assertEquals(16.0, adapter.alpha());
            assertEquals(1.0, adapter.scale());

            String base = "model.layers.0.self_attn.q_proj.weight";
            Optional<LoraAdapter.LoraDelta> delta = adapter.deltaFor(base);
            assertTrue(delta.isPresent());
            assertEquals(16, delta.get().loraA().shape().first());
            assertEquals(2048, delta.get().loraA().shape().last());
            assertEquals(2048, delta.get().loraB().shape().first());
            assertEquals(16, delta.get().loraB().shape().last());

            // This adapter has no modules_to_save and doesn't target embeddings/lm_head.
            assertFalse(adapter.deltaFor("model.embed_tokens.weight").isPresent());
            assertFalse(adapter.deltaFor("lm_head.weight").isPresent());
        }
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
