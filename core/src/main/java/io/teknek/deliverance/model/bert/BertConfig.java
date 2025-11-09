package io.teknek.deliverance.model.bert;


import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.safetensors.Config;

import java.util.List;
import java.util.Map;

public class BertConfig extends Config {
    @JsonCreator
    public BertConfig(
            @JsonProperty("max_position_embeddings") int contextLength,
            @JsonProperty("hidden_size") int embeddingLength,
            @JsonProperty("intermediate_size") int hiddenLength,
            @JsonProperty("num_attention_heads") int numberOfHeads,
            @JsonProperty("num_hidden_layers") int numberOfLayers,
            @JsonProperty("layer_norm_eps") float layerNormEps,
            @JsonProperty("hidden_act") ActivationFunction.Type activationFunction,
            @JsonProperty("vocab_size") int vocabularySize,
            @JsonProperty("label2id") Map<String, Integer> classificationLabels,
            @JsonProperty("sep_token") Integer sepToken,
            @JsonProperty("cls_token") Integer clsToken
    ) {
        super(
                contextLength,
                embeddingLength,
                hiddenLength,
                numberOfHeads,
                numberOfHeads,
                numberOfLayers,
                layerNormEps,
                vocabularySize,
                sepToken == null ? 0 : sepToken,
                clsToken == null ? List.of(0) : List.of(clsToken),
                activationFunction,
                null,
                null,
                classificationLabels
        );
    }
}