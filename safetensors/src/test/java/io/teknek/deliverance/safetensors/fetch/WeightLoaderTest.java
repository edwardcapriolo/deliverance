package io.teknek.deliverance.safetensors.fetch;

import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class WeightLoaderTest {

    @Test
    void loadUpTest() throws IOException {
        String modelName = "Qwen3-0.6B-JQ4";
        String modelOwner = "edwardcapriolo";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        try (DefaultWeightLoader wl = new DefaultWeightLoader(f)) {
            assertTrue(wl.tensorInfoMap().containsKey("model.norm.weight"));
            assertTrue(wl.tensorInfoMap().containsKey("model.layers.1.self_attn.o_proj.weight"));
        }

        try (DefaultWeightLoader wl = new DefaultWeightLoader(f)) {
            assertTrue(wl.tensorInfoMap().containsKey("model.layers.1.self_attn.o_proj.weight"));
            assertTrue(wl.isWeightPresent("model.layers.1.self_attn.o_proj.weight"));
            assertEquals(355, wl.tensorInfoMap().size());
        }
    }
}
