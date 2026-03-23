package io.teknek.deliverance.model.gpt2;


import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.safetensors.Config;

import java.util.List;

public class Gpt2Config extends Config {

    @JsonCreator
    public Gpt2Config(
            @JsonProperty("n_ctx") int contextLength,
            @JsonProperty("n_embd") int embeddingLength,
            @JsonProperty("n_head") int numberOfHeads,
            @JsonProperty("n_layer") int numberOfLayers,
            @JsonProperty("layer_norm_epsilon") float layerNormEps,
            @JsonProperty("vocab_size") int vocabularySize,
            @JsonProperty("bos_token_id") int bosToken,
            @JsonProperty("eos_token_id") int eosToken
    ) {
        super(
                contextLength,
                embeddingLength,
                embeddingLength * 4,
                numberOfHeads,
                numberOfHeads,
                numberOfLayers,
                layerNormEps,
                vocabularySize,
                bosToken,
                List.of(eosToken),
                ActivationFunction.Type.GELU,
                null,
                null
        );
    }
}