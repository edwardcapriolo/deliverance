package io.teknek.deliverance.model.embeddinggemma2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Typed configuration for the EmbeddingGemma 2 text encoder.
 *
 * <p>This class deliberately models the nested Hugging Face text configuration
 * before any model execution is implemented. Layer-specific head dimensions,
 * attention types, and RoPE parameters are retained so later execution code
 * cannot accidentally collapse the local and global attention paths.</p>
 */
public final class EmbeddingGemma2TextConfig {
    public final int vocabSize;
    public final int hiddenSize;
    public final int intermediateSize;
    public final int numHiddenLayers;
    public final int numAttentionHeads;
    public final int numKeyValueHeads;
    public final int headDim;
    public final String hiddenActivation;
    public final int maxPositionEmbeddings;
    public final float rmsNormEps;
    public final int padTokenId;
    public final int bosTokenId;
    public final List<Integer> eosTokenId;
    public final int slidingWindow;
    public final int embeddingDim;
    public final int hiddenSizePerLayerInput;
    public final List<String> layerTypes;
    public final Map<String, Map<String, Object>> ropeParameters;
    public final Map<String, Map<String, Object>> perLayerConfig;

    /**
     * Creates a typed text configuration from the nested Hugging Face JSON map.
     *
     * @param source nested {@code text_config} object from {@code config.json}
     */
    public EmbeddingGemma2TextConfig(Map<String, Object> source) {
        Map<String, Object> config = new LinkedHashMap<>(requireMap(source, "text_config"));
        this.vocabSize = intValue(config, "vocab_size");
        this.hiddenSize = intValue(config, "hidden_size");
        this.intermediateSize = intValue(config, "intermediate_size");
        this.numHiddenLayers = intValue(config, "num_hidden_layers");
        this.numAttentionHeads = intValue(config, "num_attention_heads");
        this.numKeyValueHeads = intValue(config, "num_key_value_heads");
        this.headDim = intValue(config, "head_dim");
        this.hiddenActivation = stringValue(config, "hidden_activation");
        this.maxPositionEmbeddings = intValue(config, "max_position_embeddings");
        this.rmsNormEps = floatValue(config, "rms_norm_eps");
        this.padTokenId = intValue(config, "pad_token_id");
        this.bosTokenId = intValue(config, "bos_token_id");
        this.eosTokenId = intList(config.get("eos_token_id"));
        this.slidingWindow = intValue(config, "sliding_window");
        this.embeddingDim = intValue(config, "embedding_dim");
        this.hiddenSizePerLayerInput = intValue(config, "hidden_size_per_layer_input");
        this.layerTypes = immutableStringList(config.get("layer_types"), "layer_types");
        this.ropeParameters = immutableMapOfMaps(config.get("rope_parameters"), "rope_parameters");
        this.perLayerConfig = immutableMapOfMaps(config.get("per_layer_config"), "per_layer_config");
    }

    /**
     * Returns the configured head dimension for a layer type.
     *
     * @param layerType layer type such as {@code sliding_attention} or
     *                  {@code full_attention}
     * @return head dimension for the layer type
     */
    public int headDimFor(String layerType) {
        Map<String, Object> layer = perLayerConfig.get(layerType);
        return layer == null ? headDim : intValue(layer, "head_dim");
    }

    /** Returns the head width for a concrete layer, including zero-padded real checkpoint keys. */
    public int headDimFor(int layerIndex) {
        Map<String, Object> layer = layerConfigFor(layerIndex);
        return layer == null ? headDim : intValue(layer, "head_dim");
    }

    /**
     * Returns the configured key/value head count for a layer type.
     *
     * @param layerType layer type such as {@code sliding_attention} or
     *                  {@code full_attention}
     * @return key/value head count for the layer type
     */
    public int keyValueHeadsFor(String layerType) {
        Map<String, Object> layer = perLayerConfig.get(layerType);
        return layer == null ? numKeyValueHeads : intValue(layer, "num_key_value_heads");
    }

    /** Returns the key/value head count for a concrete layer. */
    public int keyValueHeadsFor(int layerIndex) {
        Map<String, Object> layer = layerConfigFor(layerIndex);
        return layer == null ? numKeyValueHeads : intValue(layer, "num_key_value_heads");
    }

    private Map<String, Object> layerConfigFor(int layerIndex) {
        Map<String, Object> layer = perLayerConfig.get(String.format("%02d", layerIndex));
        if (layer == null) {
            layer = perLayerConfig.get(Integer.toString(layerIndex));
        }
        return layer;
    }

    private static Map<String, Object> requireMap(Map<String, Object> source, String name) {
        if (source == null) {
            throw new IllegalArgumentException("Missing " + name + " configuration");
        }
        return source;
    }

    private static int intValue(Map<String, Object> source, String name) {
        Object value = source.get(name);
        if (!(value instanceof Number number)) {
            throw new IllegalArgumentException("Missing or non-numeric " + name + " in EmbeddingGemma 2 config");
        }
        return number.intValue();
    }

    private static float floatValue(Map<String, Object> source, String name) {
        Object value = source.get(name);
        if (!(value instanceof Number number)) {
            throw new IllegalArgumentException("Missing or non-numeric " + name + " in EmbeddingGemma 2 config");
        }
        return number.floatValue();
    }

    private static String stringValue(Map<String, Object> source, String name) {
        Object value = source.get(name);
        if (!(value instanceof String string)) {
            throw new IllegalArgumentException("Missing or non-string " + name + " in EmbeddingGemma 2 config");
        }
        return string;
    }

    private static List<Integer> intList(Object value) {
        if (value instanceof Number number) {
            return List.of(number.intValue());
        }
        if (value instanceof List<?> values) {
            List<Integer> result = new ArrayList<>(values.size());
            for (Object item : values) {
                if (!(item instanceof Number number)) {
                    throw new IllegalArgumentException("Non-numeric eos_token_id in EmbeddingGemma 2 config");
                }
                result.add(number.intValue());
            }
            return List.copyOf(result);
        }
        throw new IllegalArgumentException("Missing eos_token_id in EmbeddingGemma 2 config");
    }

    private static List<String> immutableStringList(Object value, String name) {
        if (!(value instanceof List<?> values)) {
            throw new IllegalArgumentException("Missing or non-list " + name + " in EmbeddingGemma 2 config");
        }
        List<String> result = new ArrayList<>(values.size());
        for (Object item : values) {
            if (!(item instanceof String string)) {
                throw new IllegalArgumentException("Non-string item in " + name + " of EmbeddingGemma 2 config");
            }
            result.add(string);
        }
        return List.copyOf(result);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Map<String, Object>> immutableMapOfMaps(Object value, String name) {
        if (!(value instanceof Map<?, ?> values)) {
            throw new IllegalArgumentException("Missing or non-map " + name + " in EmbeddingGemma 2 config");
        }
        Map<String, Map<String, Object>> result = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : values.entrySet()) {
            if (!(entry.getKey() instanceof String key) || !(entry.getValue() instanceof Map<?, ?> nested)) {
                throw new IllegalArgumentException("Invalid entry in " + name + " of EmbeddingGemma 2 config");
            }
            result.put(key, Collections.unmodifiableMap(new LinkedHashMap<>((Map<String, Object>) nested)));
        }
        return Collections.unmodifiableMap(result);
    }
}
