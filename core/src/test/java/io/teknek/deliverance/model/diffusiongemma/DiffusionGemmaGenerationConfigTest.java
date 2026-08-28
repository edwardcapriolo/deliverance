package io.teknek.deliverance.model.diffusiongemma;

import com.fasterxml.jackson.databind.exc.ValueInstantiationException;
import com.fasterxml.jackson.databind.JsonMappingException;
import io.teknek.deliverance.JsonUtils;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DiffusionGemmaGenerationConfigTest {

    @Test
    void testGenerationConfigInterface() throws Exception {
        String json = """
                {
                  "max_length": 128,
                  "max_new_tokens": 64,
                  "cache_implementation": "dynamic",
                  "pad_token_id": 0,
                  "eos_token_id": 1
                }
                """;

        DiffusionGemmaGenerationConfig config = JsonUtils.om.readValue(json, DiffusionGemmaGenerationConfig.class);

        assertEquals(128, config.maxLength);
        assertEquals(64, config.maxNewTokens);
        assertEquals("dynamic", config.cacheImplementation);
        assertEquals(0, config.padTokenId);
        assertEquals(1, config.eosTokenId);
    }

    @Test
    void testBadDiffusionGenerationConfigParameterization() {
        List<String> badFields = List.of("do_sample", "num_beams", "num_beam_groups", "temperature", "top_k",
                "top_p", "repetition_penalty", "no_repeat_ngram_size", "encoder_no_repeat_ngram_size",
                "length_penalty", "early_stopping", "num_return_sequences", "foo");

        for (String field : badFields) {
            JsonMappingException thrown = assertThrows(JsonMappingException.class,
                    () -> JsonUtils.om.readValue("{\"" + field + "\": 1}", DiffusionGemmaGenerationConfig.class),
                    "field=" + field);
            org.junit.jupiter.api.Assertions.assertTrue(thrown.getMessage().contains(field), "field=" + field);
        }
    }

    @Test
    void testSaveLoadGenerationConfig() throws Exception {
        String original = """
                {
                  "max_new_tokens": 64,
                  "sampler_config": {"entropy_bound": 0.1},
                  "t_min": 0.4,
                  "t_max": 0.8,
                  "stability_threshold": 1,
                  "confidence_threshold": 0.005,
                  "eos_token_id": [1, 2]
                }
                """;

        DiffusionGemmaGenerationConfig config = JsonUtils.om.readValue(original, DiffusionGemmaGenerationConfig.class);
        String serialized = JsonUtils.om.writeValueAsString(config);
        DiffusionGemmaGenerationConfig loaded = JsonUtils.om.readValue(serialized, DiffusionGemmaGenerationConfig.class);

        assertEquals(64, loaded.maxNewTokens);
        assertEquals(0.1f, loaded.samplerConfig.entropyBound, 0.0f);
        assertEquals(0.4f, loaded.tMin, 0.0f);
        assertEquals(0.8f, loaded.tMax, 0.0f);
        assertEquals(1, loaded.stabilityThreshold);
        assertEquals(0.005f, loaded.confidenceThreshold, 0.0f);
        assertInstanceOf(List.class, loaded.eosTokenId);
        assertEquals(List.of(1, 2), loaded.eosTokenId);
    }

    @Test
    void defaultsMatchHuggingFaceGenerationDefaults() {
        DiffusionGemmaGenerationConfig config = new DiffusionGemmaGenerationConfig();

        assertEquals(256, config.maxLength);
        assertEquals(48, config.maxDenoisingSteps);
        assertEquals(0.1f, config.samplerConfig.entropyBound, 0.0f);
        assertEquals(0.4f, config.tMin, 0.0f);
        assertEquals(0.8f, config.tMax, 0.0f);
        assertEquals(1, config.stabilityThreshold);
        assertEquals(0.005f, config.confidenceThreshold, 0.0f);
    }

    @Test
    void rejectsInvalidDiffusionValues() {
        assertThrows(IllegalArgumentException.class, () -> new EntropyBoundSamplerConfig(0.0f));
        assertThrows(IllegalArgumentException.class, () -> new DiffusionGemmaGenerationConfig(null, -1, null,
                null, null, null, null, null, null, null, null, null, null));
        assertThrows(IllegalArgumentException.class, () -> new DiffusionGemmaGenerationConfig(null, null, 0,
                null, null, null, null, null, null, null, null, null, null));
        assertThrows(IllegalArgumentException.class, () -> new DiffusionGemmaGenerationConfig(null, null, null,
                null, 0.8f, 0.4f, null, null, null, null, null, null, null));
    }
}
