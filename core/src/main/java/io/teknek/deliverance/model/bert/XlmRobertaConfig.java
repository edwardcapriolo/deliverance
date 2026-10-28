package io.teknek.deliverance.model.bert;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.teknek.deliverance.math.ActivationFunction;

import java.util.Map;

/** Configuration for the XLM-RoBERTa encoder family. */
public class XlmRobertaConfig extends BertConfig {
    @JsonCreator
    public XlmRobertaConfig(
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
            @JsonProperty("bos_token_id") Integer bosTokenId,
            @JsonProperty("eos_token_id") Integer eosTokenId
    ) {
        super(contextLength, embeddingLength, hiddenLength, numberOfHeads, numberOfLayers, layerNormEps,
                activationFunction, vocabularySize, typeVocabSize, padTokenId, positionEmbeddingType,
                hiddenDropoutProb, attentionProbsDropoutProb, isDecoder, useCache, classificationLabels,
                eosTokenId, bosTokenId);
    }
}
