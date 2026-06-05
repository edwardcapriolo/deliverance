package io.teknek.deliverance.model;

public enum GenerationInputKey {
    INPUT_IDS("input_ids"),
    DECODER_INPUT_IDS("decoder_input_ids"),
    POSITION_IDS("position_ids"),
    DECODER_POSITION_IDS("decoder_position_ids"),
    ATTENTION_MASK("attention_mask"),
    DECODER_ATTENTION_MASK("decoder_attention_mask");

    private final String externalName;

    GenerationInputKey(String externalName) {
        this.externalName = externalName;
    }

    public String externalName() {
        return externalName;
    }
}
