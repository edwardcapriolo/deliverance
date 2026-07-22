package io.teknek.deliverance.model.granitemoehybrid;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.safetensors.Config;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class GraniteMoeHybridConfig extends Config {

    public final float attentionDropout;
    public final int sharedIntermediateSize;
    public final int numLocalExperts;
    public final int numExpertsPerToken;
    public final boolean outputRouterLogits;
    public final float routerAuxLossCoef;
    public final List<String> layerTypes;
    public final String positionEmbeddingType;
    public final int mambaNHeads;
    public final int mambaNGroups;
    public final int mambaDState;
    public final int mambaDHead;
    public final int mambaDConv;
    public final int mambaExpand;
    public final int mambaChunkSize;
    public final boolean mambaConvBias;
    public final boolean mambaProjBias;
    public final float logitsScaling;

    @JsonCreator
    public GraniteMoeHybridConfig(
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
            @JsonProperty("rope_scaling") Map<String, Object> ropeScaling,
            @JsonProperty("attention_dropout") Float attentionDropout,
            @JsonProperty("embedding_multiplier") Float embeddingMultiplier,
            @JsonProperty("attention_multiplier") Float attentionMultiplier,
            @JsonProperty("residual_multiplier") Float residualMultiplier,
            @JsonProperty("logits_scaling") Float logitsScaling,
            @JsonProperty("shared_intermediate_size") Integer sharedIntermediateSize,
            @JsonProperty("num_local_experts") Integer numLocalExperts,
            @JsonProperty("num_experts_per_tok") Integer numExpertsPerToken,
            @JsonProperty("output_router_logits") Boolean outputRouterLogits,
            @JsonProperty("router_aux_loss_coef") Float routerAuxLossCoef,
            @JsonProperty("layer_types") List<String> layerTypes,
            @JsonProperty("position_embedding_type") String positionEmbeddingType,
            @JsonProperty("mamba_n_heads") Integer mambaNHeads,
            @JsonProperty("mamba_n_groups") Integer mambaNGroups,
            @JsonProperty("mamba_d_state") Integer mambaDState,
            @JsonProperty("mamba_d_head") Object mambaDHead,
            @JsonProperty("mamba_d_conv") Integer mambaDConv,
            @JsonProperty("mamba_expand") Integer mambaExpand,
            @JsonProperty("mamba_chunk_size") Integer mambaChunkSize,
            @JsonProperty("mamba_conv_bias") Boolean mambaConvBias,
            @JsonProperty("mamba_proj_bias") Boolean mambaProjBias,
            @JsonProperty("architectures") List<String> architectures
    ) {
        super(
                contextLength,
                embeddingLength,
                hiddenLength,
                numberOfHeads,
                numberOfKeyValueHeads == null ? numberOfHeads : numberOfKeyValueHeads,
                numberOfLayers,
                layerNormEps,
                vocabularySize,
                bosToken == null ? 0 : bosToken,
                toIntList(eosToken),
                activationFunction,
                ropeTheta(ropeTheta, ropeParameters),
                ropeScaling,
                null,
                embeddingLength / numberOfHeads,
                null,
                null,
                residualMultiplier,
                attentionMultiplier,
                embeddingMultiplier,
                logitsScaling,
                architectures
        );
        this.attentionDropout = attentionDropout == null ? 0.0f : attentionDropout;
        this.sharedIntermediateSize = sharedIntermediateSize == null ? 1024 : sharedIntermediateSize;
        this.numLocalExperts = numLocalExperts == null ? 8 : numLocalExperts;
        this.numExpertsPerToken = numExpertsPerToken == null ? 2 : numExpertsPerToken;
        this.outputRouterLogits = Boolean.TRUE.equals(outputRouterLogits);
        this.routerAuxLossCoef = routerAuxLossCoef == null ? 0.001f : routerAuxLossCoef;
        this.layerTypes = layerTypes == null ? defaultLayerTypes(numberOfLayers) : List.copyOf(layerTypes);
        this.positionEmbeddingType = positionEmbeddingType;
        this.mambaNHeads = mambaNHeads == null ? 128 : mambaNHeads;
        this.mambaNGroups = mambaNGroups == null ? 1 : mambaNGroups;
        this.mambaDState = mambaDState == null ? 256 : mambaDState;
        this.mambaExpand = mambaExpand == null ? 2 : mambaExpand;
        this.mambaDHead = mambaDHead(mambaDHead, this.mambaExpand, embeddingLength, this.mambaNHeads);
        this.mambaDConv = mambaDConv == null ? 4 : mambaDConv;
        this.mambaChunkSize = mambaChunkSize == null ? 256 : mambaChunkSize;
        this.mambaConvBias = mambaConvBias == null || mambaConvBias;
        this.mambaProjBias = Boolean.TRUE.equals(mambaProjBias);
        this.logitsScaling = logitsScaling == null ? 1.0f : logitsScaling;
    }

    public boolean denseAttentionOnly() {
        return this.numLocalExperts == 0 && this.numExpertsPerToken == 0
                && this.layerTypes.stream().allMatch("attention"::equals);
    }

    private static Double ropeTheta(Double ropeTheta, Map<String, Object> ropeParameters) {
        if (ropeParameters != null && ropeParameters.get("rope_theta") instanceof Number n) {
            return n.doubleValue();
        }
        return ropeTheta == null ? 10_000.0 : ropeTheta;
    }

    private static List<String> defaultLayerTypes(int layers) {
        List<String> types = new ArrayList<>(layers);
        for (int i = 0; i < layers; i++) {
            types.add("mamba");
        }
        return List.copyOf(types);
    }

    private static int mambaDHead(Object value, int mambaExpand, int hiddenSize, int mambaNHeads) {
        if (value instanceof Number n) {
            return n.intValue();
        }
        return (mambaExpand * hiddenSize) / mambaNHeads;
    }

    private static List<Integer> toIntList(Object eosToken) {
        if (eosToken == null) {
            return List.of();
        }
        if (eosToken instanceof Number n) {
            return List.of(n.intValue());
        }
        if (eosToken instanceof List<?> values) {
            return values.stream().map(v -> ((Number) v).intValue()).toList();
        }
        throw new IllegalArgumentException("Unsupported eos_token_id: " + eosToken);
    }

}
