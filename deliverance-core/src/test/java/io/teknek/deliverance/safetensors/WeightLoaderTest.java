package io.teknek.deliverance.safetensors;

import io.teknek.deliverance.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.AbstractTensor;
import org.junit.jupiter.api.Test;

import java.io.File;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class WeightLoaderTest {

    @Test
    void loadUpTest(){
        String modelName = "TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
        String modelOwner = "tjake";
        ModelFetcher fetch = new ModelFetcher(modelOwner, modelName);
        File f = fetch.maybeDownload();
        DefaultWeightLoader wl = new DefaultWeightLoader(f);
        assertTrue(wl.tensorInfoMap().containsKey("model.norm.weight"));
        assertTrue(wl.tensorInfoMap().containsKey("model.layers.1.self_attn.o_proj.weight"));
        //AbstractTensor tensor = wl.load("a-test-name");
        //assertEquals("5", tensor.shape());

    }
}