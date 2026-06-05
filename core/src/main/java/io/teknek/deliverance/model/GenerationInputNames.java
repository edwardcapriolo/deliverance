package io.teknek.deliverance.model;

import io.teknek.deliverance.safetensors.Config;

/**
 * Names the model-input fields used by generation preparation.
 *
 * <p>Decoder-only models use the base input names. Encoder-decoder models use decoder-side names for generation.</p>
 */
public record GenerationInputNames(GenerationInputKey inputIdsKey, GenerationInputKey positionIdsKey,
        GenerationInputKey attentionMaskKey) {

    public static GenerationInputNames forConfig(Config config) {
        if (config.isEncoderDecoder) {
            return new GenerationInputNames(GenerationInputKey.DECODER_INPUT_IDS,
                    GenerationInputKey.DECODER_POSITION_IDS, GenerationInputKey.DECODER_ATTENTION_MASK);
        }
        return new GenerationInputNames(GenerationInputKey.INPUT_IDS,
                GenerationInputKey.POSITION_IDS, GenerationInputKey.ATTENTION_MASK);
    }
}
