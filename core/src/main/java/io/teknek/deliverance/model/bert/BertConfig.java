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
            @JsonProperty("type_vocab_size") Integer typeVocabSize,
            @JsonProperty("pad_token_id") Integer padTokenId,
            @JsonProperty("position_embedding_type") String positionEmbeddingType,
            @JsonProperty("hidden_dropout_prob") Float hiddenDropoutProb,
            @JsonProperty("attention_probs_dropout_prob") Float attentionProbsDropoutProb,
            @JsonProperty("is_decoder") Boolean isDecoder,
            @JsonProperty("use_cache") Boolean useCache,
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
        this.typeVocabSize = typeVocabSize == null ? 2 : typeVocabSize;
        this.padTokenId = padTokenId == null ? 0 : padTokenId;
        this.positionEmbeddingType = positionEmbeddingType == null ? "absolute" : positionEmbeddingType;
        this.hiddenDropoutProb = hiddenDropoutProb == null ? 0.1f : hiddenDropoutProb;
        this.attentionProbsDropoutProb = attentionProbsDropoutProb == null ? 0.1f : attentionProbsDropoutProb;
        this.isDecoder = isDecoder != null && isDecoder;
        this.useCache = useCache == null || useCache;
    }

    public final int typeVocabSize;
    public final int padTokenId;
    public final String positionEmbeddingType;
    public final float hiddenDropoutProb;
    public final float attentionProbsDropoutProb;
    public final boolean isDecoder;
    public final boolean useCache;

    public BertConfig(int contextLength, int embeddingLength, int hiddenLength, int numberOfHeads,
            int numberOfLayers, float layerNormEps, ActivationFunction.Type activationFunction, int vocabularySize,
            Map<String, Integer> classificationLabels, Integer sepToken, Integer clsToken) {
        this(contextLength, embeddingLength, hiddenLength, numberOfHeads, numberOfLayers, layerNormEps,
                activationFunction, vocabularySize, null, null, null, null, null, null, null, classificationLabels,
                sepToken, clsToken);
    }
}
