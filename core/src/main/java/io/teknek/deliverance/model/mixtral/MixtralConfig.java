package io.teknek.deliverance.model.mixtral;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.safetensors.Config;

import java.util.List;

public class MixtralConfig extends Config {

    public final int numberOfExperts;

    public final int numberOfExpertsPerToken;

    @JsonCreator
    public MixtralConfig(
            @JsonProperty("max_position_embeddings") int contextLength,
            @JsonProperty("hidden_size") int embeddingLength,
            @JsonProperty("intermediate_size") int hiddenLength,
            @JsonProperty("num_attention_heads") int numberOfHeads,
            @JsonProperty("num_key_value_heads") int numberOfKeyValueHeads,
            @JsonProperty("num_hidden_layers") int numberOfLayers,
            @JsonProperty("rms_norm_eps") float layerNormEps,
            @JsonProperty("vocab_size") int vocabularySize,
            @JsonProperty("bos_token_id") int bosToken,
            @JsonProperty("eos_token_id") int eosToken,
            @JsonProperty("rope_theta") Double ropeTheta,
            @JsonProperty("hidden_act") ActivationFunction.Type activationFunction,
            @JsonProperty("num_local_experts") int numberOfExperts,
            @JsonProperty("num_experts_per_tok") int numberOfExpertsPerToken
    ) {
        super(
                contextLength,
                embeddingLength,
                hiddenLength,
                numberOfHeads,
                numberOfKeyValueHeads,
                numberOfLayers,
                layerNormEps,
                vocabularySize,
                bosToken,
                List.of(eosToken),
                activationFunction,
                ropeTheta,
                null
        );

        this.numberOfExperts = numberOfExperts;
        this.numberOfExpertsPerToken = numberOfExpertsPerToken;
    }
}