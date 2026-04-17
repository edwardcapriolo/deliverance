package io.teknek.deliverance.model.gemma3;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.safetensors.Config;

import java.util.List;
import java.util.Map;

public class Gemma3Config extends Config {
    @JsonCreator
    public Gemma3Config(
            @JsonProperty("max_position_embeddings") int contextLength,
            @JsonProperty("hidden_size") int embeddingLength,
            @JsonProperty("intermediate_size") int hiddenLength,
            @JsonProperty("num_attention_heads") int numberOfHeads,
            @JsonProperty("num_key_value_heads") int numberOfKeyValueHeads,
            @JsonProperty("num_hidden_layers") int numberOfLayers,
            @JsonProperty("rms_norm_eps") float layerNormEps,
            @JsonProperty("vocab_size") int vocabularySize,
            @JsonProperty("bos_token_id") int bosToken,
            @JsonProperty("eos_token_id") Object eosTokens,
            @JsonProperty("hidden_activation") ActivationFunction.Type activationFunction,
            @JsonProperty("rope_theta") Double ropeFreqsTheta,
            @JsonProperty("rope_scaling") Map<String, Object> ropeScaling,
            @JsonProperty("head_dim") Integer headDim,
            @JsonProperty("final_logit_softcapping") Float finalLogitSoftCapping,
            @JsonProperty("attn_logit_softcapping") Float attnLogitSoftCapping
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
                eosTokens instanceof List ? (List<Integer>) eosTokens : List.of((Integer) eosTokens),
                activationFunction,
                ropeFreqsTheta == null ? 10000.0 : ropeFreqsTheta,
                ropeScaling,
                headDim,
                finalLogitSoftCapping,
                attnLogitSoftCapping
        );
    }
}