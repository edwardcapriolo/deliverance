package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class WeightLoaderTest {

    @Test
    void loadUpTest() throws IOException {
        String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        DefaultWeightLoader wl = new DefaultWeightLoader(f);
        assertTrue(wl.tensorInfoMap().containsKey("model.norm.weight"));
        assertTrue(wl.tensorInfoMap().containsKey("model.layers.1.self_attn.o_proj.weight"));

        wl.close();
        assertEquals(0, wl.tensorInfoMap().size());
        wl.loadWeights();
        assertTrue(wl.tensorInfoMap().containsKey("model.layers.1.self_attn.o_proj.weight"));
        assertTrue(wl.isWeightPresent("model.layers.1.self_attn.o_proj.weight"));
        assertEquals(355, wl.tensorInfoMap().size());
        wl.close();
    }
}