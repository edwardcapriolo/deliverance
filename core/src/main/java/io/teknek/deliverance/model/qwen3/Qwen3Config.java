package io.teknek.deliverance.model.qwen3;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.safetensors.Config;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Qwen3Config extends Config {
    public final boolean useSlidingWindow;
    public final Integer slidingWindow;
    public final int maxWindowLayers;
    public final List<String> layerTypes;
    public final float attentionDropout;
    public final Map<String, Object> ropeParameters;

    @JsonCreator
    public Qwen3Config(
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
                ConfigSupport.toIntList(eosToken),
                activationFunction,
                ropeTheta(ropeTheta, ropeParameters),
                null,
                null,
                headDim == null ? embeddingLength / numberOfHeads : headDim,
                null,
                null,
                null,
                null,
                null,
                null,
                architectures
        );
        this.useSlidingWindow = Boolean.TRUE.equals(useSlidingWindow);
        this.slidingWindow = this.useSlidingWindow ? slidingWindow : null;
        this.maxWindowLayers = maxWindowLayers == null ? 28 : maxWindowLayers;
        this.layerTypes = layerTypes == null ? defaultLayerTypes(numberOfLayers, this.slidingWindow, this.maxWindowLayers) : List.copyOf(layerTypes);
        this.attentionDropout = attentionDropout == null ? 0.0f : attentionDropout;
        this.ropeParameters = ropeParameters;
    }

    private static Double ropeTheta(Double ropeTheta, Map<String, Object> ropeParameters) {
        if (ropeParameters != null && ropeParameters.get("rope_theta") instanceof Number n) {
            return n.doubleValue();
        }
        return ropeTheta == null ? 1_000_000.0 : ropeTheta;
    }

    private static List<String> defaultLayerTypes(int layers, Integer slidingWindow, int maxWindowLayers) {
        List<String> types = new ArrayList<>(layers);
        for (int i = 0; i < layers; i++) {
            types.add(slidingWindow != null && i >= maxWindowLayers ? "sliding_attention" : "full_attention");
        }
        return List.copyOf(types);
    }

    static final class ConfigSupport {
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
}
