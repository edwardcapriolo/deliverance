package io.teknek.deliverance.integration;

import io.teknek.deliverance.safetensors.DefaultWeightLoader;
import io.teknek.deliverance.safetensors.fetch.ModelFetcher;
import io.teknek.deliverance.tensor.TensorInfo;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class Gemma4WeightLayoutIT {

    @Test
    public void e2bUsesTiedOutputHeadFromTextEmbeddings() {
        ModelFetcher fetch = new ModelFetcher("google", "gemma-4-E2B-it");
        try (DefaultWeightLoader loader = new DefaultWeightLoader(fetch.pathForModel().toFile())) {
            assertFalse(loader.isWeightPresent("lm_head.weight"), "Gemma4 E2B should use tied embeddings, not root lm_head.weight");
            assertFalse(loader.isWeightPresent("language_model.lm_head.weight"), "Gemma4 E2B should use tied embeddings, not language_model.lm_head.weight");
            assertFalse(loader.isWeightPresent("model.language_model.lm_head.weight"), "Gemma4 E2B should use tied embeddings, not model.language_model.lm_head.weight");

            String embed = "model.language_model.embed_tokens.weight";
            assertTrue(loader.isWeightPresent(embed), "Missing text embedding weight used as tied output head");
            TensorInfo info = loader.tensorInfoMap().get(embed);
            assertArrayEquals(new int[]{262144, 1536}, info.shape);
        }
    }
}
