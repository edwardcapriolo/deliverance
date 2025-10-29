package io.teknek.deliverance.safetensors.fetch;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;

public class ModelFetcherTest {
    @Test
    void downloadAModel(){
        ModelFetcher fetch = new ModelFetcher("tjake", "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4");
        File f = fetch.maybeDownload();
        Assertions.assertTrue(f.exists() && f.isDirectory());
    }
}
