package io.teknek.deliverance.fetch;

import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.File;

public class ModelFetcherTest {
    @Test
    void aTest(){
        String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        Assertions.assertTrue(f.exists() && f.isDirectory());
    }
}
