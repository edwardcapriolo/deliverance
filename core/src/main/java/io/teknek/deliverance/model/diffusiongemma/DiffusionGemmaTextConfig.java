package io.teknek.deliverance.model.diffusiongemma;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.teknek.deliverance.math.ActivationFunction;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class DiffusionGemmaTextConfig {
    public static final String MODEL_TYPE = "diffusion_gemma_text";

    public final int vocabSize;
    public final int hiddenSize;
    public final int intermediateSize;
    public final int numHiddenLayers;
    public final int numAttentionHeads;
    public final int numKeyValueHeads;
    public final int headDim;
    public final ActivationFunction.Type hiddenActivation;
    public final int maxPositionEmbeddings;
    public final float initializerRange;
    public final float rmsNormEps;
    public final Integer padTokenId;
    public final Object eosTokenId;
    public final Integer bosTokenId;
    public final boolean tieWordEmbeddings;
    public final Map<LayerType, RopeParameters> ropeParameters;
    public final boolean attentionBias;
    public final float attentionDropout;
    public final int slidingWindow;
    public final List<LayerType> layerTypes;
    public final float finalLogitSoftcapping;
    public final BidirectionalAttention useBidirectionalAttention;
    public final Integer numGlobalKeyValueHeads;
    public final int globalHeadDim;
    public final Integer numExperts;
    public final Integer topKExperts;
    public final Integer moeIntermediateSize;
    public final boolean causal;

    public DiffusionGemmaTextConfig() {
        this(null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null,
                null, null, null, null, null, null, null, null, null, null, null);
    }

    @JsonCreator
    public DiffusionGemmaTextConfig(
            @JsonProperty("vocab_size") Integer vocabSize,
            @JsonProperty("hidden_size") Integer hiddenSize,
            @JsonProperty("intermediate_size") Integer intermediateSize,
            @JsonProperty("num_hidden_layers") Integer numHiddenLayers,
            @JsonProperty("num_attention_heads") Integer numAttentionHeads,
            @JsonProperty("num_key_value_heads") Integer numKeyValueHeads,
            @JsonProperty("head_dim") Integer headDim,
            @JsonProperty("hidden_activation") String hiddenActivation,
            @JsonProperty("max_position_embeddings") Integer maxPositionEmbeddings,
            @JsonProperty("initializer_range") Float initializerRange,
            @JsonProperty("rms_norm_eps") Float rmsNormEps,
            @JsonProperty("pad_token_id") Integer padTokenId,
            @JsonProperty("eos_token_id") Object eosTokenId,
            @JsonProperty("bos_token_id") Integer bosTokenId,
            @JsonProperty("tie_word_embeddings") Boolean tieWordEmbeddings,
            @JsonProperty("rope_parameters") Map<String, Map<String, Object>> ropeParameters,
            @JsonProperty("attention_bias") Boolean attentionBias,
            @JsonProperty("attention_dropout") Float attentionDropout,
            @JsonProperty("sliding_window") Integer slidingWindow,
            @JsonProperty("layer_types") List<String> layerTypes,
            @JsonProperty("final_logit_softcapping") Float finalLogitSoftcapping,
            @JsonProperty("use_bidirectional_attention") BidirectionalAttention useBidirectionalAttention,
            @JsonProperty("num_global_key_value_heads") Integer numGlobalKeyValueHeads,
            @JsonProperty("global_head_dim") Integer globalHeadDim,
            @JsonProperty("num_experts") Integer numExperts,
            @JsonProperty("top_k_experts") Integer topKExperts,
            @JsonProperty("moe_intermediate_size") Integer moeIntermediateSize) {
        this.vocabSize = defaultInt(vocabSize, 262_144);
        this.hiddenSize = defaultInt(hiddenSize, 2304);
        this.intermediateSize = defaultInt(intermediateSize, 9216);
        this.numHiddenLayers = defaultInt(numHiddenLayers, 30);
        this.numAttentionHeads = defaultInt(numAttentionHeads, 8);
        this.numKeyValueHeads = defaultInt(numKeyValueHeads, 4);
        this.headDim = defaultInt(headDim, 256);
        this.hiddenActivation = activation(defaultString(hiddenActivation, "gelu_pytorch_tanh"));
        this.maxPositionEmbeddings = defaultInt(maxPositionEmbeddings, 131_072);
        this.initializerRange = defaultFloat(initializerRange, 0.02f);
        this.rmsNormEps = defaultFloat(rmsNormEps, 1.0e-6f);
        this.padTokenId = padTokenId == null ? 0 : padTokenId;
        this.eosTokenId = eosTokenId == null ? 1 : eosTokenId;
        this.bosTokenId = bosTokenId == null ? 2 : bosTokenId;
        this.tieWordEmbeddings = tieWordEmbeddings == null || tieWordEmbeddings;
        this.attentionBias = attentionBias != null && attentionBias;
        this.attentionDropout = defaultFloat(attentionDropout, 0.0f);
        this.useBidirectionalAttention = useBidirectionalAttention;
        this.causal = useBidirectionalAttention != BidirectionalAttention.ALL;
        int rawSlidingWindow = defaultInt(slidingWindow, 512);
        this.slidingWindow = useBidirectionalAttention == BidirectionalAttention.ALL ? rawSlidingWindow / 2 + 1 : rawSlidingWindow;
        this.layerTypes = Collections.unmodifiableList(normalizeLayerTypes(layerTypes, this.numHiddenLayers));
        this.finalLogitSoftcapping = defaultFloat(finalLogitSoftcapping, 30.0f);
        this.numGlobalKeyValueHeads = numGlobalKeyValueHeads;
        this.globalHeadDim = defaultInt(globalHeadDim, 512);
        this.numExperts = numExperts;
        this.topKExperts = topKExperts;
        this.moeIntermediateSize = moeIntermediateSize;
        this.ropeParameters = Collections.unmodifiableMap(normalizeRopeParameters(ropeParameters));
    }

    public enum LayerType {
        SLIDING_ATTENTION("sliding_attention"),
        FULL_ATTENTION("full_attention");

        public final String value;

        LayerType(String value) {
            this.value = value;
        }

        @JsonCreator
        public static LayerType fromJson(String value) {
            for (LayerType type : values()) {
                if (type.value.equals(value) || type.name().equalsIgnoreCase(value)) {
                    return type;
                }
            }
            throw new IllegalArgumentException("Unknown DiffusionGemma layer type " + value);
        }
    }

    public enum BidirectionalAttention {
        ALL("all"),
        VISION("vision");

        public final String value;

        BidirectionalAttention(String value) {
            this.value = value;
        }

        @JsonCreator
        public static BidirectionalAttention fromJson(String value) {
            if (value == null) {
                return null;
            }
            for (BidirectionalAttention mode : values()) {
                if (mode.value.equals(value) || mode.name().equalsIgnoreCase(value)) {
                    return mode;
                }
            }
            throw new IllegalArgumentException("Unknown DiffusionGemma bidirectional attention mode " + value);
        }
    }

    public enum RopeType {
        DEFAULT("default"),
        PROPORTIONAL("proportional");

        public final String value;

        RopeType(String value) {
            this.value = value;
        }

        @JsonCreator
        public static RopeType fromJson(String value) {
            for (RopeType type : values()) {
                if (type.value.equals(value) || type.name().equalsIgnoreCase(value)) {
                    return type;
                }
            }
            throw new IllegalArgumentException("Unknown DiffusionGemma RoPE type " + value);
        }
    }

    public record RopeParameters(RopeType ropeType, double ropeTheta, Double partialRotaryFactor) {
    }

    private static List<LayerType> normalizeLayerTypes(List<String> raw, int layers) {
        List<LayerType> result = new ArrayList<>();
        if (raw == null) {
            for (int i = 0; i < layers; i++) {
                result.add(((i + 1) % 6) == 0 ? LayerType.FULL_ATTENTION : LayerType.SLIDING_ATTENTION);
            }
        } else {
            raw.stream().map(LayerType::fromJson).forEach(result::add);
        }
        if (!result.isEmpty() && result.get(result.size() - 1) != LayerType.FULL_ATTENTION) {
            result.set(result.size() - 1, LayerType.FULL_ATTENTION);
        }
        return result;
    }

    private static Map<LayerType, RopeParameters> normalizeRopeParameters(Map<String, Map<String, Object>> raw) {
        Map<LayerType, RopeParameters> result = new LinkedHashMap<>();
        Map<String, Map<String, Object>> source = raw == null ? Map.of(
                "sliding_attention", Map.of("rope_type", "default", "rope_theta", 10_000.0),
                "full_attention", Map.of("rope_type", "proportional", "partial_rotary_factor", 0.25, "rope_theta", 1_000_000.0)
        ) : raw;
        source.forEach((key, value) -> result.put(LayerType.fromJson(key), rope(value)));
        return result;
    }

    private static RopeParameters rope(Map<String, Object> value) {
        RopeType ropeType = RopeType.fromJson(Objects.toString(value.getOrDefault("rope_type", "default")));
        double theta = number(value.get("rope_theta"), 10_000.0).doubleValue();
        Object partial = value.get("partial_rotary_factor");
        return new RopeParameters(ropeType, theta, partial == null ? null : ((Number) partial).doubleValue());
    }

    private static ActivationFunction.Type activation(String value) {
        return ActivationFunction.Type.valueOf(value.toUpperCase());
    }

    private static Number number(Object value, double fallback) {
        return value instanceof Number number ? number : fallback;
    }

    private static int defaultInt(Integer value, int fallback) {
        return value == null ? fallback : value;
    }

    private static float defaultFloat(Float value, float fallback) {
        return value == null ? fallback : value;
    }

    private static String defaultString(String value, String fallback) {
        return value == null ? fallback : value;
    }
}
