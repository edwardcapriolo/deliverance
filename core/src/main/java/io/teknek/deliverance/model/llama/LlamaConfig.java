package io.teknek.deliverance.model.llama;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;


import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.safetensors.Config;

import java.util.List;
import java.util.Map;

public class LlamaConfig extends Config {

    @JsonCreator
    public LlamaConfig(
            @JsonProperty("max_position_embeddings") int contextLength,
            @JsonProperty("hidden_size") int embeddingLength,
            @JsonProperty("intermediate_size") int hiddenLength,
            @JsonProperty("num_attention_heads") int numberOfHeads,
            @JsonProperty("num_key_value_heads") int numberOfKeyValueHeads,
            @JsonProperty("num_hidden_layers") int numberOfLayers,
            @JsonProperty("rms_norm_eps") float layerNormEps,
            @JsonProperty("vocab_size") int vocabularySize,
            @JsonProperty("bos_token_id") int bosToken,
            @JsonProperty("eos_token_id") Object eosToken,
            @JsonProperty("hidden_act") ActivationFunction.Type activationFunction,
            @JsonProperty("rope_theta") Double ropeFreqsTheta,
            @JsonProperty("rope_scaling") Map<String, String> ropeScaling
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
                eosToken instanceof List<?> ? (List<Integer>) eosToken : List.of((Integer) eosToken),
                activationFunction,
                ropeFreqsTheta == null ? 10000.0 : ropeFreqsTheta,
                ropeScaling == null || !("linear".equals(ropeScaling.get("rope_type"))) ? 1.0 : Double.parseDouble(ropeScaling.get("factor"))
        );
    }
}