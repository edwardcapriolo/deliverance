package io.teknek.reckoning.safetensors;

import io.teknek.reckoning.tensor.AbstractTensor;
import io.teknek.reckoning.tensor.impl.FloatBufferTensor;
import io.dropwizard.metrics5.MetricRegistry;
import io.teknek.reckoning.safetensors.fetch.LoraAdapterModelFetcher;
import io.teknek.reckoning.tensor2.Lighter;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class LoraAdapterTest {

    @TempDir
    Path tempDir;

    private final Lighter lighter = new Lighter();

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
            adapter.setLighter(lighter);
            assertEquals(4, adapter.rank());
            assertEquals(8.0, adapter.alpha());
            assertEquals(2.0, adapter.scale());

            try (LoraAdapter.LoraDelta delta = adapter.deltaFor(base).orElseThrow()) {
                assertEquals(rankDim, delta.loraA().shape().first());
                assertEquals(inFeatures, delta.loraA().shape().last());
                assertEquals(outFeatures, delta.loraB().shape().first());
                assertEquals(rankDim, delta.loraB().shape().last());
            }

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

    @Test
    void loadFailsFastWhenLoraATensorHasNoMatchingLoraBTensor() throws IOException {
        writeAdapterConfig(tempDir, 4, 8.0, "q_proj");

        String base = "model.layers.0.self_attn.q_proj.weight";
        SafeTensorWriter.write(tempDir.resolve(LoraAdapter.SAFETENSORS_FILE_NAME), Map.of(),
                Map.of(LoraTensorNames.loraA(base), new FloatBufferTensor(4, 8)));

        assertThrows(IllegalStateException.class, () -> LoraAdapter.load(tempDir.toFile()));
    }

    @Test
    void missingAdapterWeightsAreReportedAsUncheckedIo() throws IOException {
        writeAdapterConfig(tempDir, 4, 8.0, "q_proj");

        assertThrows(java.io.UncheckedIOException.class, () -> LoraAdapter.load(tempDir.toFile()));
    }

    @Test
    void recordsParseAndTensorLoadMetrics() throws IOException {
        writeAdapterConfig(tempDir, 2, 4.0, "q_proj");
        String base = "model.layers.0.self_attn.q_proj.weight";
        SafeTensorWriter.write(tempDir.resolve(LoraAdapter.SAFETENSORS_FILE_NAME), Map.of(), Map.of(
                LoraTensorNames.loraA(base), new FloatBufferTensor(2, 3),
                LoraTensorNames.loraB(base), new FloatBufferTensor(4, 2)));
        MetricRegistry metrics = new MetricRegistry();

        try (LoraAdapter adapter = LoraAdapter.load(tempDir.toFile(), metrics)) {
            adapter.setLighter(lighter);
            try (LoraAdapter.LoraDelta delta = adapter.deltaFor(base).orElseThrow()) {
            assertEquals(1, metrics.timer("loraadapter.parse_header").getCount());
            assertEquals(1, metrics.timer("loraadapter.load_tensor").getCount());
            }
        }
    }

    /**
     * Integration test against a real, small, publicly published PEFT LoRA adapter for
     * Llama-3.2-1B (rank 16, ~45MB, no {@code modules_to_save} bloat -- verified before use,
     * see the parent plan and {@code LoraTensorNamesTest}'s naming-convention test, both
     * checked against this adapter's actual safetensors header). Requires network access.
     */
    @Test
    @Tag("large-model")
    void fromPretrainedDownloadsAndParsesARealPublishedAdapter() {
        // Uses the default ~/.deliverance cache (like the project's other HF-backed tests,
        // e.g. LeafModelExample) rather than an isolated temp dir, so repeated local/CI runs
        // reuse the download instead of re-fetching ~45MB every time.
        LoraAdapterModelFetcher fetcher = new LoraAdapterModelFetcher("bunnycore", "Llama-3.2-1b-chatml-lora_model");

        try (LoraAdapter adapter = LoraAdapter.fromPretrained(fetcher)) {
            adapter.setLighter(lighter);
            assertEquals(16, adapter.rank());
            assertEquals(16.0, adapter.alpha());
            assertEquals(1.0, adapter.scale());

            String base = "model.layers.0.self_attn.q_proj.weight";
            try (LoraAdapter.LoraDelta delta = adapter.deltaFor(base).orElseThrow()) {
                assertEquals(16, delta.loraA().shape().first());
                assertEquals(2048, delta.loraA().shape().last());
                assertEquals(2048, delta.loraB().shape().first());
                assertEquals(16, delta.loraB().shape().last());
            }

            // This adapter has no modules_to_save and doesn't target embeddings/lm_head.
            assertFalse(adapter.deltaFor("model.embed_tokens.weight").isPresent());
            assertFalse(adapter.deltaFor("lm_head.weight").isPresent());
        }
    }

    @Test
    @Tag("large-model")
    void fromPretrainedDownloadsAndParsesNemotronLinearSpecSubfolderAdapter() {
        LoraAdapterModelFetcher fetcher = new LoraAdapterModelFetcher("nvidia", "Nemotron-Labs-Diffusion-3B",
                "linear_spec_lora", true);

        try (LoraAdapter adapter = LoraAdapter.fromPretrained(fetcher)) {
            adapter.setLighter(lighter);
            assertEquals(128, adapter.rank());
            assertEquals(512.0, adapter.alpha());
            assertEquals(4.0, adapter.scale());

            try (LoraAdapter.LoraDelta delta = adapter.deltaFor("encoder.layers.0.self_attn.o_proj.weight").orElseThrow()) {
                assertEquals(128, delta.loraA().shape().first());
                assertEquals(4096, delta.loraA().shape().last());
                assertEquals(3072, delta.loraB().shape().first());
                assertEquals(128, delta.loraB().shape().last());
            }

            assertFalse(adapter.deltaFor("encoder.layers.0.self_attn.q_proj.weight").isPresent());
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
