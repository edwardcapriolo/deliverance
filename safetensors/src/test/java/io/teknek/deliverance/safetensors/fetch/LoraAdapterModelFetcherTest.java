package io.teknek.deliverance.safetensors.fetch;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LoraAdapterModelFetcherTest {
    @TempDir
    Path tempDir;

    @Test
    void topLevelFetcherKeepsAdapterFilesAndDropsUnrelatedFiles() {
        LoraAdapterModelFetcher fetcher = new LoraAdapterModelFetcher("owner", "adapter");

        List<String> selected = fetcher.filesToDownload(List.of(
                "README.md", "adapter_config.json", "adapter_model.safetensors", "pytorch_model.bin",
                "nested/ignored.json"), true);

        assertEquals(List.of("README.md", "adapter_config.json", "adapter_model.safetensors"), selected);
        fetcher.setBaseDir(tempDir);
        assertEquals(tempDir.resolve("owner_adapter"), fetcher.pathForModel());
    }

    @Test
    void subfolderFetcherNormalizesPathAndMapsFilesToLocalRoot() {
        LoraAdapterModelFetcher fetcher = new LoraAdapterModelFetcher("owner", "adapter", "/linear_spec_lora/", true);

        fetcher.setBaseDir(tempDir);
        assertEquals(tempDir.resolve("owner_adapter_linear_spec_lora"), fetcher.pathForModel());
        assertEquals(List.of(
                        "linear_spec_lora/README.md",
                        "linear_spec_lora/adapter_config.json",
                        "linear_spec_lora/adapter_model.safetensors"),
                fetcher.filesToDownload(List.of(
                        "README.md",
                        "linear_spec_lora/README.md",
                        "linear_spec_lora/adapter_config.json",
                        "linear_spec_lora/adapter_model.safetensors",
                        "linear_spec_lora/weights.bin"), true));
    }

    @Test
    void subfolderAdapterRequiresBothNonEmptyFilesLocally() throws Exception {
        LoraAdapterModelFetcher fetcher = new LoraAdapterModelFetcher("owner", "adapter", "subfolder", true);
        fetcher.setBaseDir(tempDir);
        Path model = fetcher.pathForModel();
        Files.createDirectories(model);

        assertFalse(fetcher.isLocallyComplete(ModelFetcher.FetchPolicy.FULL_MODEL, model));

        Files.writeString(model.resolve("adapter_config.json"), "{}");
        Files.createFile(model.resolve("adapter_model.safetensors"));
        assertFalse(fetcher.isLocallyComplete(ModelFetcher.FetchPolicy.FULL_MODEL, model));

        Files.writeString(model.resolve("adapter_model.safetensors"), "weights");
        assertTrue(fetcher.isLocallyComplete(ModelFetcher.FetchPolicy.FULL_MODEL, model));
    }

    @Test
    void booleanConstructorCanDisableSubfolderHandling() {
        LoraAdapterModelFetcher fetcher = new LoraAdapterModelFetcher("owner", "adapter", "subfolder", false);

        fetcher.setBaseDir(tempDir);
        assertEquals(tempDir.resolve("owner_adapter"), fetcher.pathForModel());
        assertEquals(List.of("adapter_config.json"),
                fetcher.filesToDownload(List.of("subfolder/adapter_config.json", "adapter_config.json"), true));
    }
}
