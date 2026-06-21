package io.teknek.deliverance.model.hf;

import io.teknek.deliverance.safetensors.Config;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Ports feasible text-model checks from HF {@code ConfigTester}. */
public interface HfConfigTesterMixinPort extends HfTextCommonTestAdapter {
    @Test
    default void hfConfigTesterCommonProperties() {
        Config config = loadTinyConfig(writeTinyCheckpoint("hf-config-common-properties", 20_000));

        assertTrue(config.contextLength > 0);
        assertTrue(config.embeddingLength > 0);
        assertTrue(config.vocabularySize > 0);
        assertTrue(config.numberOfLayers > 0);
        assertFalse(config.eosTokens.isEmpty());
        assertFalse(config.architectures.isEmpty());
    }

    @Test
    default void hfConfigTesterConfigJsonRoundTrip() throws Exception {
        Config first = loadTinyConfig(writeTinyCheckpoint("hf-config-json-round-trip", 20_001));
        Config second = roundTripConfig(first);

        assertEquals(first.contextLength, second.contextLength);
        assertEquals(first.embeddingLength, second.embeddingLength);
        assertEquals(first.vocabularySize, second.vocabularySize);
        assertEquals(first.numberOfLayers, second.numberOfLayers);
        assertEquals(first.eosTokens, second.eosTokens);
        assertEquals(first.architectures, second.architectures);
        assertModelSpecificConfigRoundTrip(first, second);
    }

    @Test
    default void hfConfigTesterFromAndSavePretrainedEquivalent() throws Exception {
        Path modelDir = writeTinyCheckpoint("hf-config-save-source", 20_002);
        Config source = loadTinyConfig(modelDir);
        Config reloaded = roundTripConfig(source);

        assertEquals(source.contextLength, reloaded.contextLength);
        assertEquals(source.embeddingLength, reloaded.embeddingLength);
        assertEquals(source.vocabularySize, reloaded.vocabularySize);
        assertModelSpecificConfigRoundTrip(source, reloaded);
    }
}
