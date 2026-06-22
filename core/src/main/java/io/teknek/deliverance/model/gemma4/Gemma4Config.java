package io.teknek.deliverance.model.gemma4;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.safetensors.Config;
import net.jafama.FastMath;

import java.util.*;

public class Gemma4Config extends Config {
    public final Integer slidingWindow;
    public final List<String> layerTypes;
    public final Integer numGlobalKeyValueHeads;
    public final Integer globalHeadDim;
    public final boolean attentionKEqV;
    public final int numKvSharedLayers;
    public final boolean enableMoeBlock;
    public final boolean useDoubleWideMlp;
    public final String useBidirectionalAttention;
    public final Integer numExperts;
    public final Integer topKExperts;
    public final Integer moeIntermediateSize;
    public final Integer vocabSizePerLayerInput;
    public final Integer hiddenSizePerLayerInput;
    public final int defaultHeadDim;
    public final int defaultKeyValueHeads;
    public final Map<String, Map<String, Object>> ropeParametersByLayerType;
    public final Map<String, float[]> ropeInvFreqsByLayerType;
    public final Map<String, Integer> rotaryDimensionsByLayerType;

    @JsonCreator
    @SuppressWarnings("unchecked")
    public Gemma4Config(
            @JsonProperty("text_config") Map<String, Object> textConfig,
            @JsonProperty("architectures") List<String> architectures,
            @JsonProperty("eos_token_id") Object rootEosTokens
    ) {
        super(
                intValue(normalizeTextConfig(textConfig), "max_position_embeddings"),
                intValue(normalizeTextConfig(textConfig), "hidden_size"),
                intValue(normalizeTextConfig(textConfig), "intermediate_size"),
                intValue(normalizeTextConfig(textConfig), "num_attention_heads"),
                Math.max(intValue(normalizeTextConfig(textConfig), "num_key_value_heads"), intValue(normalizeTextConfig(textConfig), "num_global_key_value_heads", intValue(normalizeTextConfig(textConfig), "num_key_value_heads"))),
                intValue(normalizeTextConfig(textConfig), "num_hidden_layers"),
                floatValue(normalizeTextConfig(textConfig), "rms_norm_eps"),
                intValue(normalizeTextConfig(textConfig), "vocab_size"),
                intValue(normalizeTextConfig(textConfig), "bos_token_id"),
                toIntList(rootEosTokens != null ? rootEosTokens : normalizeTextConfig(textConfig).get("eos_token_id")),
                activation(normalizeTextConfig(textConfig)),
                ropeTheta((Map<String, Object>) mapValue(mapValue(normalizeTextConfig(textConfig), "rope_parameters"), "sliding_attention"), 10000.0),
                null,
                null,
                Math.max(intValue(normalizeTextConfig(textConfig), "head_dim"), intValue(normalizeTextConfig(textConfig), "global_head_dim", intValue(normalizeTextConfig(textConfig), "head_dim"))),
                floatValueObject(normalizeTextConfig(textConfig), "final_logit_softcapping"),
                floatValueObject(normalizeTextConfig(textConfig), "attn_logit_softcapping"),
                null,
                null,
                null,
                null,
                architectures
        );

        Map<String, Object> normalizedTextConfig = normalizeTextConfig(textConfig);
        this.slidingWindow = intValueObject(normalizedTextConfig, "sliding_window");
        this.layerTypes = Collections.unmodifiableList(new ArrayList<>((List<String>) normalizedTextConfig.get("layer_types")));
        this.defaultHeadDim = intValue(normalizedTextConfig, "head_dim");
        this.defaultKeyValueHeads = intValue(normalizedTextConfig, "num_key_value_heads");
        this.numGlobalKeyValueHeads = intValueObject(normalizedTextConfig, "num_global_key_value_heads");
        this.globalHeadDim = intValueObject(normalizedTextConfig, "global_head_dim");
        this.attentionKEqV = booleanValue(normalizedTextConfig, "attention_k_eq_v");
        this.numKvSharedLayers = intValue(normalizedTextConfig, "num_kv_shared_layers", 0);
        this.enableMoeBlock = booleanValue(normalizedTextConfig, "enable_moe_block");
        this.useDoubleWideMlp = booleanValue(normalizedTextConfig, "use_double_wide_mlp");
        this.useBidirectionalAttention = stringValueObject(normalizedTextConfig, "use_bidirectional_attention");
        this.numExperts = intValueObject(normalizedTextConfig, "num_experts");
        this.topKExperts = intValueObject(normalizedTextConfig, "top_k_experts");
        this.moeIntermediateSize = firstNonNullInt(normalizedTextConfig, "moe_intermediate_size", "expert_intermediate_size");
        this.vocabSizePerLayerInput = intValueObject(normalizedTextConfig, "vocab_size_per_layer_input");
        this.hiddenSizePerLayerInput = intValueObject(normalizedTextConfig, "hidden_size_per_layer_input");

        Map<String, Object> ropeParameters = mapValue(normalizedTextConfig, "rope_parameters");
        Map<String, Map<String, Object>> typedRopeParams = new LinkedHashMap<>();
        Map<String, float[]> typedFreqs = new LinkedHashMap<>();
        Map<String, Integer> typedRotaryDims = new LinkedHashMap<>();
        for (String layerType : this.layerTypes.stream().distinct().toList()) {
            Map<String, Object> params = mapValue(ropeParameters, layerType);
            typedRopeParams.put(layerType, params);
            int rotaryDim = evenFloor(getLayerHeadDim(layerType));
            typedRotaryDims.put(layerType, rotaryDim);
            typedFreqs.put(layerType, precomputeRopeFreqs(layerType, params));
        }
        this.ropeParametersByLayerType = Collections.unmodifiableMap(typedRopeParams);
        this.ropeInvFreqsByLayerType = Collections.unmodifiableMap(typedFreqs);
        this.rotaryDimensionsByLayerType = Collections.unmodifiableMap(typedRotaryDims);
    }

    private static Map<String, Object> normalizeTextConfig(Map<String, Object> source) {
        Map<String, Object> config = new LinkedHashMap<>(source);
        int layers = intValue(config, "num_hidden_layers");
        if (config.get("layer_types") == null) {
            List<String> layerTypes = new ArrayList<>();
            for (int i = 0; i < layers; i++) {
                layerTypes.add(((i + 1) % 6) == 0 ? "full_attention" : "sliding_attention");
            }
            config.put("layer_types", layerTypes);
        }
        List<String> layerTypes = new ArrayList<>((List<String>) config.get("layer_types"));
        if (!layerTypes.isEmpty() && !Objects.equals(layerTypes.getLast(), "full_attention")) {
            layerTypes.set(layerTypes.size() - 1, "full_attention");
            config.put("layer_types", layerTypes);
        }
        if (config.get("rope_parameters") == null) {
            config.put("rope_parameters", Map.of(
                    "sliding_attention", Map.of("rope_type", "default", "rope_theta", 10_000.0),
                    "full_attention", Map.of("rope_type", "proportional", "partial_rotary_factor", 0.25, "rope_theta", 1_000_000.0)
            ));
        }
        return config;
    }

    public int getLayerHeadDim(String layerType) {
        if (Objects.equals(layerType, "full_attention") && globalHeadDim != null) {
            return globalHeadDim;
        }
        return defaultHeadDim;
    }

    /**
     * Gemma4 full-attention layers do not always use {@code num_global_key_value_heads}. They only
     * switch to the global KV head count when {@code attention_k_eq_v} is enabled. Otherwise they
     * continue to use the regular text KV head count with the larger {@code global_head_dim}.
     */
    public int getLayerKeyValueProjectionHeads(String layerType) {
        if (Objects.equals(layerType, "full_attention") && attentionKEqV && numGlobalKeyValueHeads != null) {
            return numGlobalKeyValueHeads;
        }
        return defaultKeyValueHeads;
    }

    public int getLayerQueryProjectionLength(String layerType) {
        return numberOfHeads * getLayerHeadDim(layerType);
    }

    public int getLayerKeyValueProjectionLength(String layerType) {
        return getLayerHeadDim(layerType) * getLayerKeyValueProjectionHeads(layerType);
    }

    public int getLayerKvLength(String layerType) {
        return getLayerKeyValueProjectionLength(layerType);
    }

    /**
     * Gemma4 proportional RoPE keeps the full head width but zero-fills the non-rotary inverse
     * frequencies, which yields identity rotation (`cos=1`, `sin=0`) for the remaining pairs.
     */
    float[] precomputeRopeFreqs(String layerType, Map<String, Object> ropeParams) {
        int headDim = evenFloor(getLayerHeadDim(layerType));
        int halfDim = headDim / 2;
        float[] freqs = new float[halfDim];
        String ropeType = stringValue(ropeParams, "rope_type");
        double theta = ropeTheta(ropeParams, 10000.0);
        double factor = doubleValue(ropeParams, "factor", 1.0);

        if (Objects.equals(ropeType, "proportional")) {
            double partialRotaryFactor = doubleValue(ropeParams, "partial_rotary_factor", 1.0);
            int ropeAngles = Math.min(halfDim, (int) Math.floor(partialRotaryFactor * headDim / 2.0));
            for (int i = 0; i < ropeAngles; i++) {
                freqs[i] = (float) ((1.0 / FastMath.pow(theta, (2.0 * i) / headDim)) / factor);
            }
        } else {
            for (int i = 0; i < freqs.length; i++) {
                freqs[i] = (float) ((1.0 / FastMath.pow(theta, (2.0 * i) / headDim)) / factor);
            }
        }

        return freqs;
    }

    public int getLayerHiddenLength(int layerIndex) {
        boolean isKvSharedLayer = layerIndex >= (numberOfLayers - numKvSharedLayers) && numKvSharedLayers > 0;
        return useDoubleWideMlp && isKvSharedLayer ? hiddenLength * 2 : hiddenLength;
    }

    public boolean useAllBidirectionalAttention() {
        return Objects.equals(useBidirectionalAttention, "all");
    }

    public int getSharedKvSourceLayer(int layerIndex) {
        int firstSharedLayer = numberOfLayers - numKvSharedLayers;
        if (layerIndex < firstSharedLayer || numKvSharedLayers <= 0) {
            return -1;
        }
        String layerType = layerTypes.get(layerIndex);
        for (int i = firstSharedLayer - 1; i >= 0; i--) {
            if (Objects.equals(layerType, layerTypes.get(i))) {
                return i;
            }
        }
        throw new IllegalArgumentException("No non-shared source layer for " + layerIndex + " type " + layerType);
    }

    public boolean storesSharedKvState(int layerIndex) {
        int firstSharedLayer = numberOfLayers - numKvSharedLayers;
        int limit = Math.max(firstSharedLayer, 0);
        String layerType = layerTypes.get(layerIndex);
        for (int i = limit - 1; i >= 0; i--) {
            if (Objects.equals(layerType, layerTypes.get(i))) {
                return i == layerIndex;
            }
        }
        return false;
    }

    private static ActivationFunction.Type activation(Map<String, Object> textConfig) {
        return ActivationFunction.Type.valueOf(stringValue(textConfig, "hidden_activation").toUpperCase());
    }

    private static Map<String, Object> mapValue(Map<String, Object> source, String key) {
        Object value = source.get(key);
        if (value == null) {
            return Collections.emptyMap();
        }
        return (Map<String, Object>) value;
    }

    private static String stringValue(Map<String, Object> source, String key) {
        return String.valueOf(source.get(key));
    }

    private static String stringValueObject(Map<String, Object> source, String key) {
        Object value = source.get(key);
        return value == null ? null : String.valueOf(value);
    }

    private static int intValue(Map<String, Object> source, String key) {
        return ((Number) source.get(key)).intValue();
    }

    private static int intValue(Map<String, Object> source, String key, int defaultValue) {
        Object value = source.get(key);
        return value == null ? defaultValue : ((Number) value).intValue();
    }

    private static Integer intValueObject(Map<String, Object> source, String key) {
        Object value = source.get(key);
        return value == null ? null : ((Number) value).intValue();
    }

    private static Integer firstNonNullInt(Map<String, Object> source, String firstKey, String secondKey) {
        Integer first = intValueObject(source, firstKey);
        return first != null ? first : intValueObject(source, secondKey);
    }

    private static boolean booleanValue(Map<String, Object> source, String key) {
        Object value = source.get(key);
        return value instanceof Boolean && (Boolean) value;
    }

    private static float floatValue(Map<String, Object> source, String key) {
        return ((Number) source.get(key)).floatValue();
    }

    private static Float floatValueObject(Map<String, Object> source, String key) {
        Object value = source.get(key);
        return value == null ? null : ((Number) value).floatValue();
    }

    private static double doubleValue(Map<String, Object> source, String key, double defaultValue) {
        Object value = source.get(key);
        return value == null ? defaultValue : ((Number) value).doubleValue();
    }

    private static double ropeTheta(Map<String, Object> ropeParameters, double defaultValue) {
        Object value = ropeParameters.get("rope_theta");
        return value == null ? defaultValue : ((Number) value).doubleValue();
    }

    private static int evenFloor(int value) {
        return (value & 1) == 0 ? value : value - 1;
    }


    @SuppressWarnings("unchecked")
    private static List<Integer> toIntList(Object eosTokens) {
        if (eosTokens instanceof List<?>) {
            List<Integer> result = new ArrayList<>();
            for (Object token : (List<Object>) eosTokens) {
                result.add(((Number) token).intValue());
            }
            return result;
        }
        return List.of(((Number) eosTokens).intValue());
    }
}
