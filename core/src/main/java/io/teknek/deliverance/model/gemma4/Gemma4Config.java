package io.teknek.deliverance.model.gemma4;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.math.VectorMathUtils;
import io.teknek.deliverance.safetensors.Config;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class Gemma4Config extends Config {
    public final Integer slidingWindow;
    public final List<String> layerTypes;
    public final Integer numGlobalKeyValueHeads;
    public final Integer globalHeadDim;
    public final boolean attentionKEqV;
    public final int numKvSharedLayers;
    public final boolean enableMoeBlock;
    public final boolean useDoubleWideMlp;
    public final Integer numExperts;
    public final Integer topKExperts;
    public final Integer moeIntermediateSize;
    public final Integer vocabSizePerLayerInput;
    public final Integer hiddenSizePerLayerInput;
    public final int defaultHeadDim;
    public final int defaultKeyValueHeads;
    public final Map<String, Map<String, Object>> ropeParametersByLayerType;
    public final Map<String, float[][]> ropeFreqsByLayerType;
    public final Map<String, Integer> rotaryDimensionsByLayerType;

    @JsonCreator
    @SuppressWarnings("unchecked")
    public Gemma4Config(
            @JsonProperty("text_config") Map<String, Object> textConfig,
            @JsonProperty("architectures") List<String> architectures,
            @JsonProperty("eos_token_id") Object rootEosTokens
    ) {
        super(
                intValue(textConfig, "max_position_embeddings"),
                intValue(textConfig, "hidden_size"),
                intValue(textConfig, "intermediate_size"),
                intValue(textConfig, "num_attention_heads"),
                Math.max(intValue(textConfig, "num_key_value_heads"), intValue(textConfig, "num_global_key_value_heads", intValue(textConfig, "num_key_value_heads"))),
                intValue(textConfig, "num_hidden_layers"),
                floatValue(textConfig, "rms_norm_eps"),
                intValue(textConfig, "vocab_size"),
                intValue(textConfig, "bos_token_id"),
                toIntList(rootEosTokens != null ? rootEosTokens : textConfig.get("eos_token_id")),
                activation(textConfig),
                ropeTheta((Map<String, Object>) mapValue(mapValue(textConfig, "rope_parameters"), "sliding_attention"), 10000.0),
                null,
                null,
                Math.max(intValue(textConfig, "head_dim"), intValue(textConfig, "global_head_dim", intValue(textConfig, "head_dim"))),
                floatValueObject(textConfig, "final_logit_softcapping"),
                floatValueObject(textConfig, "attn_logit_softcapping"),
                null,
                null,
                null,
                null,
                architectures
        );

        this.slidingWindow = intValueObject(textConfig, "sliding_window");
        this.layerTypes = Collections.unmodifiableList(new ArrayList<>((List<String>) textConfig.get("layer_types")));
        this.defaultHeadDim = intValue(textConfig, "head_dim");
        this.defaultKeyValueHeads = intValue(textConfig, "num_key_value_heads");
        this.numGlobalKeyValueHeads = intValueObject(textConfig, "num_global_key_value_heads");
        this.globalHeadDim = intValueObject(textConfig, "global_head_dim");
        this.attentionKEqV = booleanValue(textConfig, "attention_k_eq_v");
        this.numKvSharedLayers = intValue(textConfig, "num_kv_shared_layers", 0);
        this.enableMoeBlock = booleanValue(textConfig, "enable_moe_block");
        this.useDoubleWideMlp = booleanValue(textConfig, "use_double_wide_mlp");
        this.numExperts = intValueObject(textConfig, "num_experts");
        this.topKExperts = intValueObject(textConfig, "top_k_experts");
        this.moeIntermediateSize = intValueObject(textConfig, "moe_intermediate_size");
        this.vocabSizePerLayerInput = intValueObject(textConfig, "vocab_size_per_layer_input");
        this.hiddenSizePerLayerInput = intValueObject(textConfig, "hidden_size_per_layer_input");

        Map<String, Object> ropeParameters = mapValue(textConfig, "rope_parameters");
        Map<String, Map<String, Object>> typedRopeParams = new LinkedHashMap<>();
        Map<String, float[][]> typedFreqs = new LinkedHashMap<>();
        Map<String, Integer> typedRotaryDims = new LinkedHashMap<>();
        for (String layerType : this.layerTypes.stream().distinct().toList()) {
            Map<String, Object> params = mapValue(ropeParameters, layerType);
            typedRopeParams.put(layerType, params);
            int headDim = getLayerHeadDim(layerType);
            double partialRotaryFactor = doubleValue(params, "partial_rotary_factor", 1.0);
            int rotaryDim = Math.max(2, evenFloor((int) Math.round(headDim * partialRotaryFactor)));
            typedRotaryDims.put(layerType, rotaryDim);
            typedFreqs.put(layerType, VectorMathUtils.precomputeFreqsCis(rotaryDim, contextLength, ropeTheta(params, 10000.0), 1.0));
        }
        this.ropeParametersByLayerType = Collections.unmodifiableMap(typedRopeParams);
        this.ropeFreqsByLayerType = Collections.unmodifiableMap(typedFreqs);
        this.rotaryDimensionsByLayerType = Collections.unmodifiableMap(typedRotaryDims);
    }

    public int getLayerHeadDim(String layerType) {
        if (Objects.equals(layerType, "full_attention") && globalHeadDim != null) {
            return globalHeadDim;
        }
        return defaultHeadDim;
    }

    public int getLayerKeyValueHeads(String layerType) {
        if (Objects.equals(layerType, "full_attention") && numGlobalKeyValueHeads != null) {
            return numGlobalKeyValueHeads;
        }
        return defaultKeyValueHeads;
    }

    public int getLayerKvLength(String layerType) {
        return getLayerHeadDim(layerType) * getLayerKeyValueHeads(layerType);
    }

    public int getLayerHiddenLength(int layerIndex) {
        boolean isKvSharedLayer = layerIndex >= (numberOfLayers - numKvSharedLayers) && numKvSharedLayers > 0;
        return useDoubleWideMlp && isKvSharedLayer ? hiddenLength * 2 : hiddenLength;
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
