package io.teknek.deliverance.safetensors.fetch;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

public class ModelFetcherTest {
    @Test
    void downloadAModel(){
        //given a modelname
        ModelFetcher fetch = new ModelFetcher("tjake", "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4");
        //when i try maybe to downlad the model
        File f = fetch.maybeDownload();
        //then the directory exists
        Assertions.assertTrue(f.exists() && f.isDirectory());
    }

    @Test
    void includesChatTemplateJinjaInDownloads() {
        ModelFetcher fetch = new ModelFetcher("google", "gemma-4-E2B-it");
        List<String> files = fetch.filesToDownload(List.of(
                "config.json",
                "chat_template.jinja",
                "tokenizer.json",
                "tokenizer_config.json",
                "model.safetensors"
        ), true);
        Assertions.assertTrue(files.contains("chat_template.jinja"));
    }

    @Test
    void fileCompletenessUsesMissingZeroAndSizeMismatch() throws Exception {
        ModelFetcher fetch = new ModelFetcher("google", "gemma-4-E2B-it");
        Path temp = Files.createTempDirectory("fetcher-test");
        Path file = temp.resolve("tokenizer.json");
        ModelFetcher.RemoteFileMetadata metadata = new ModelFetcher.RemoteFileMetadata("tokenizer.json", 10L);

        Assertions.assertFalse(fetch.isFileComplete(file, metadata));

        Files.createFile(file);
        Assertions.assertFalse(fetch.isFileComplete(file, metadata));

        Files.write(file, new byte[9]);
        Assertions.assertFalse(fetch.isFileComplete(file, metadata));

        Files.write(file, new byte[10]);
        Assertions.assertTrue(fetch.isFileComplete(file, metadata));
    }

    @Test
    void incompleteFilesReturnsOnlyBrokenOnes() throws Exception {
        ModelFetcher fetch = new ModelFetcher("google", "gemma-4-E2B-it");
        Path temp = Files.createTempDirectory("fetcher-incomplete");
        Files.write(temp.resolve("tokenizer.json"), new byte[12]);
        Files.createFile(temp.resolve("config.json"));
        Map<String, ModelFetcher.RemoteFileMetadata> metadata = Map.of(
                "tokenizer.json", new ModelFetcher.RemoteFileMetadata("tokenizer.json", 12L),
                "config.json", new ModelFetcher.RemoteFileMetadata("config.json", 8L),
                "chat_template.jinja", new ModelFetcher.RemoteFileMetadata("chat_template.jinja", 4L)
        );

        List<String> missing = fetch.findIncompleteFiles(metadata, temp);
        Assertions.assertEquals(List.of("config.json", "chat_template.jinja").stream().sorted().toList(),
                missing.stream().sorted().toList());
    }

    @Test
    void localFullModelDirectoryCanBeUsedOffline() throws Exception {
        ModelFetcher fetch = new ModelFetcher("edward", "gemma-4-E2B-it-JQ4");
        Path temp = Files.createTempDirectory("fetcher-local-model");
        Files.writeString(temp.resolve("config.json"), "{}");
        Files.writeString(temp.resolve("tokenizer.json"), "{}");
        Files.writeString(temp.resolve("model.safetensors.index.json"), "{}");

        Assertions.assertTrue(fetch.isLocallyComplete(ModelFetcher.FetchPolicy.FULL_MODEL, temp));
    }

    @Test
    void localTokenizerDirectoryCanBeUsedOffline() throws Exception {
        ModelFetcher fetch = new ModelFetcher("google", "gemma-4-E2B-it");
        Path temp = Files.createTempDirectory("fetcher-local-tokenizer");
        Files.writeString(temp.resolve("tokenizer.json"), "{}");

        Assertions.assertTrue(fetch.isLocallyComplete(ModelFetcher.FetchPolicy.TOKENIZER_ONLY, temp));
    }
}
