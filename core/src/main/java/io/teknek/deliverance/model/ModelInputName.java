package io.teknek.deliverance.model;

/**
 * Primary model input names supported by Deliverance.
 *
 * <p>Avoids raw strings because the domain of possible values is small and known enough to extend deliberately.</p>
 */
public enum ModelInputName {
    /** Default primary input for decoder-only and encoder-only text models. */
    INPUT_IDS("input_ids"),
    /** Decoder-side token input for encoder-decoder generation models. */
    DECODER_INPUT_IDS("decoder_input_ids"),
    /** Precomputed token embeddings used instead of token ids. */
    INPUTS_EMBEDS("inputs_embeds"),
    /** Image or video pixel tensor input for vision and multimodal models. */
    PIXEL_VALUES("pixel_values"),
    /** Processed audio feature tensor input, such as log-mel features. */
    INPUT_FEATURES("input_features"),
    /** Raw audio value input, such as waveform samples. */
    INPUT_VALUES("input_values");

    private final String externalName;

    ModelInputName(String externalName) {
        this.externalName = externalName;
    }

    public String externalName() {
        return externalName;
    }
}
