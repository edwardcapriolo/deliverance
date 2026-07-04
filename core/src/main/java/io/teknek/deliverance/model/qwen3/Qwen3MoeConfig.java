package io.teknek.deliverance.model.qwen3;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.teknek.deliverance.math.ActivationFunction;

import java.util.List;
import java.util.Map;

public class Qwen3MoeConfig extends Qwen3Config {
    public final int decoderSparseStep;
    public final int moeIntermediateSize;
    public final int numExpertsPerToken;
    public final int numExperts;
    public final boolean normTopkProb;
    public final boolean outputRouterLogits;
    public final float routerAuxLossCoef;
    public final List<Integer> mlpOnlyLayers;

    @JsonCreator
    public Qwen3MoeConfig(
            @JsonProperty("max_position_embeddings") int contextLength,
            @JsonProperty("hidden_size") int embeddingLength,
            @JsonProperty("intermediate_size") int hiddenLength,
            @JsonProperty("num_attention_heads") int numberOfHeads,
            @JsonProperty("num_key_value_heads") Integer numberOfKeyValueHeads,
            @JsonProperty("num_hidden_layers") int numberOfLayers,
            @JsonProperty("rms_norm_eps") float layerNormEps,
            @JsonProperty("vocab_size") int vocabularySize,
            @JsonProperty("bos_token_id") Integer bosToken,
            @JsonProperty("eos_token_id") Object eosToken,
            @JsonProperty("hidden_act") ActivationFunction.Type activationFunction,
            @JsonProperty("rope_theta") Double ropeTheta,
            @JsonProperty("rope_parameters") Map<String, Object> ropeParameters,
            @JsonProperty("head_dim") Integer headDim,
            @JsonProperty("use_sliding_window") Boolean useSlidingWindow,
            @JsonProperty("sliding_window") Integer slidingWindow,
            @JsonProperty("max_window_layers") Integer maxWindowLayers,
            @JsonProperty("layer_types") List<String> layerTypes,
            @JsonProperty("attention_dropout") Float attentionDropout,
            @JsonProperty("architectures") List<String> architectures,
            @JsonProperty("decoder_sparse_step") Integer decoderSparseStep,
            @JsonProperty("moe_intermediate_size") Integer moeIntermediateSize,
            @JsonProperty("num_experts_per_tok") Integer numExpertsPerToken,
            @JsonProperty("num_experts") Integer numExperts,
            @JsonProperty("norm_topk_prob") Boolean normTopkProb,
            @JsonProperty("output_router_logits") Boolean outputRouterLogits,
            @JsonProperty("router_aux_loss_coef") Float routerAuxLossCoef,
            @JsonProperty("mlp_only_layers") List<Integer> mlpOnlyLayers
    ) {
        super(contextLength, embeddingLength, hiddenLength, numberOfHeads, numberOfKeyValueHeads, numberOfLayers,
                layerNormEps, vocabularySize, bosToken, eosToken, activationFunction, ropeTheta, ropeParameters,
                headDim, useSlidingWindow, slidingWindow, maxWindowLayers, layerTypes, attentionDropout, architectures);
        this.decoderSparseStep = decoderSparseStep == null ? 1 : decoderSparseStep;
        this.moeIntermediateSize = moeIntermediateSize == null ? 768 : moeIntermediateSize;
        this.numExpertsPerToken = numExpertsPerToken == null ? 8 : numExpertsPerToken;
        this.numExperts = numExperts == null ? 128 : numExperts;
        this.normTopkProb = Boolean.TRUE.equals(normTopkProb);
        this.outputRouterLogits = Boolean.TRUE.equals(outputRouterLogits);
        this.routerAuxLossCoef = routerAuxLossCoef == null ? 0.001f : routerAuxLossCoef;
        this.mlpOnlyLayers = mlpOnlyLayers == null ? List.of() : List.copyOf(mlpOnlyLayers);
        if (this.numExperts < 1) {
            throw new IllegalArgumentException("num_experts must be positive");
        }
        if (this.numExpertsPerToken < 1 || this.numExpertsPerToken > this.numExperts) {
            throw new IllegalArgumentException("num_experts_per_tok must be in [1, num_experts]");
        }
        if (this.decoderSparseStep < 1) {
            throw new IllegalArgumentException("decoder_sparse_step must be positive");
        }
    }

    public boolean sparseLayer(int layerIndex) {
        return !mlpOnlyLayers.contains(layerIndex)
                && numExperts > 0
                && (layerIndex + 1) % decoderSparseStep == 0;
    }
}
