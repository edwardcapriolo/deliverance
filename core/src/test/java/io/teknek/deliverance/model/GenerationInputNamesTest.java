package io.teknek.deliverance.model;

import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.safetensors.Config;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GenerationInputNamesTest {

    @Test
    public void decoderOnlyConfigUsesInputIds() {
        Config config = decoderOnlyConfig();

        GenerationInputNames names = GenerationInputNames.forConfig(config);

        assertFalse(config.isEncoderDecoder);
        assertEquals(GenerationInputKey.INPUT_IDS, names.inputIdsKey());
        assertEquals(GenerationInputKey.POSITION_IDS, names.positionIdsKey());
        assertEquals(GenerationInputKey.ATTENTION_MASK, names.attentionMaskKey());
    }

    @Test
    public void encoderDecoderConfigUsesDecoderInputIds() {
        Config config = encoderDecoderConfig();

        GenerationInputNames names = GenerationInputNames.forConfig(config);

        assertTrue(config.isEncoderDecoder);
        assertEquals(GenerationInputKey.DECODER_INPUT_IDS, names.inputIdsKey());
        assertEquals(GenerationInputKey.DECODER_POSITION_IDS, names.positionIdsKey());
        assertEquals(GenerationInputKey.DECODER_ATTENTION_MASK, names.attentionMaskKey());
    }

    private static Config decoderOnlyConfig() {
        return new Config(32, 16, 32, 2, 1, 1,
                1e-6f, 64, 2, List.of(1),
                ActivationFunction.Type.GELU_PYTORCH_TANH, null, null);
    }

    private static Config encoderDecoderConfig() {
        return new Config(32, 16, 32, 2, 1, 1,
                1e-6f, 64, 2, List.of(1),
                ActivationFunction.Type.GELU_PYTORCH_TANH, null, null, true);
    }
}
