package io.teknek.deliverance.model.nemotronlabsdiffusion;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.safetensors.Config;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Configuration for NVIDIA Nemotron-Labs-Diffusion models. */
public final class NemotronLabsDiffusionConfig extends Config {
    public static final String MODEL_TYPE = "nemotron_labs_diffusion";

    public final int maskTokenId;
    public final int blockSize;
    public final String dlmParadigm;
    public final Float dlmLossWeight;
    public final float arLossWeight;
    public final boolean dpVaryingMaskRatio;
    public final String attnImplementation;
    public final boolean attentionBias;
    public final boolean mlpBias;
    public final float attentionDropout;
    public final Integer slidingWindow;
    public final Map<String, Object> ropeParameters;

    @JsonCreator
    public NemotronLabsDiffusionConfig(
            @JsonProperty("max_position_embeddings") int contextLength,
            @JsonProperty("hidden_size") int embeddingLength,
            @JsonProperty("intermediate_size") int hiddenLength,
            @JsonProperty("num_attention_heads") int numberOfHeads,
            @JsonProperty("num_key_value_heads") Integer numberOfKeyValueHeads,
            @JsonProperty("num_hidden_layers") int numberOfLayers,
            @JsonProperty("rms_norm_eps") float layerNormEps,
            @JsonProperty("vocab_size") int vocabularySize,
            @JsonProperty("bos_token_id") int bosToken,
            @JsonProperty("eos_token_id") Object eosToken,
            @JsonProperty("hidden_act") String activationFunction,
            @JsonProperty("rope_parameters") Map<String, Object> ropeParameters,
            @JsonProperty("head_dim") Integer headSize,
            @JsonProperty("mask_token_id") Integer maskTokenId,
            @JsonProperty("block_size") Integer blockSize,
            @JsonProperty("dlm_paradigm") String dlmParadigm,
            @JsonProperty("dlm_loss_weight") Float dlmLossWeight,
            @JsonProperty("ar_loss_weight") Float arLossWeight,
            @JsonProperty("dp_varying_mask_ratio") Boolean dpVaryingMaskRatio,
            @JsonProperty("attn_implementation") String attnImplementation,
            @JsonProperty("attention_bias") Boolean attentionBias,
            @JsonProperty("mlp_bias") Boolean mlpBias,
            @JsonProperty("attention_dropout") Float attentionDropout,
            @JsonProperty("sliding_window") Integer slidingWindow,
            @JsonProperty("architectures") List<String> architectures) {
        super(
                contextLength,
                embeddingLength,
                hiddenLength,
                numberOfHeads,
                numberOfKeyValueHeads == null ? numberOfHeads : numberOfKeyValueHeads,
                numberOfLayers,
                layerNormEps,
                vocabularySize,
                bosToken,
                eosTokens(eosToken),
                activation(activationFunction),
                null,
                ropeParameters,
                null,
                headSize == null ? embeddingLength / numberOfHeads : headSize,
                null,
                null,
                null,
                null,
                null,
                null,
                architectures);
        this.maskTokenId = maskTokenId == null ? -1 : maskTokenId;
        this.blockSize = blockSize == null ? 32 : blockSize;
        this.dlmParadigm = dlmParadigm == null ? "bidirectional" : dlmParadigm;
        this.dlmLossWeight = dlmLossWeight;
        this.arLossWeight = arLossWeight == null ? 1.0f : arLossWeight;
        this.dpVaryingMaskRatio = dpVaryingMaskRatio != null && dpVaryingMaskRatio;
        this.attnImplementation = attnImplementation == null ? "eager" : attnImplementation;
        this.attentionBias = attentionBias != null && attentionBias;
        this.mlpBias = mlpBias != null && mlpBias;
        this.attentionDropout = attentionDropout == null ? 0.0f : attentionDropout;
        this.slidingWindow = slidingWindow;
        this.ropeParameters = ropeParameters == null ? Map.of() : Map.copyOf(ropeParameters);
    }

    private static ActivationFunction.Type activation(String value) {
        return ActivationFunction.Type.valueOf(Objects.toString(value, "silu").toUpperCase());
    }

    private static Double ropeTheta(Map<String, Object> ropeParameters) {
        Object theta = ropeParameters == null ? null : ropeParameters.get("rope_theta");
        return theta instanceof Number number ? number.doubleValue() : 1_000_000.0d;
    }

    private static List<Integer> eosTokens(Object eosToken) {
        if (eosToken instanceof List<?> list) {
            return list.stream().map(value -> ((Number) value).intValue()).toList();
        }
        if (eosToken instanceof Number number) {
            return List.of(number.intValue());
        }
        return List.of(2);
    }
}
