package io.teknek.deliverance.safetensors.fetch;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.List;

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
}
