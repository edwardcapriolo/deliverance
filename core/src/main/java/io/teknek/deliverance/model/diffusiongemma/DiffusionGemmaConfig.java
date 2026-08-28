package io.teknek.deliverance.model.diffusiongemma;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.teknek.deliverance.safetensors.Config;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class DiffusionGemmaConfig extends Config {
    public static final String MODEL_TYPE = "diffusion_gemma";

    public final DiffusionGemmaTextConfig textConfig;
    public final Map<String, Object> visionConfig;
    public final Integer boiTokenId;
    public final Integer eoiTokenId;
    public final Integer imageTokenId;
    public final float initializerRange;
    public final boolean tieWordEmbeddings;
    public final int canvasLength;

    public DiffusionGemmaConfig() {
        this(null, null, null, null, null, null, null, null);
    }

    @JsonCreator
    public DiffusionGemmaConfig(
            @JsonProperty("text_config") DiffusionGemmaTextConfig textConfig,
            @JsonProperty("vision_config") Map<String, Object> visionConfig,
            @JsonProperty("boi_token_id") Integer boiTokenId,
            @JsonProperty("eoi_token_id") Integer eoiTokenId,
            @JsonProperty("image_token_id") Integer imageTokenId,
            @JsonProperty("initializer_range") Float initializerRange,
            @JsonProperty("tie_word_embeddings") Boolean tieWordEmbeddings,
            @JsonProperty("canvas_length") Integer canvasLength) {
        super(text(textConfig).maxPositionEmbeddings, text(textConfig).hiddenSize, text(textConfig).intermediateSize,
                text(textConfig).numAttentionHeads, text(textConfig).numKeyValueHeads,
                text(textConfig).numHiddenLayers, text(textConfig).rmsNormEps, text(textConfig).vocabSize,
                text(textConfig).bosTokenId, eosTokens(text(textConfig).eosTokenId), text(textConfig).hiddenActivation,
                null, null, null, text(textConfig).headDim, text(textConfig).finalLogitSoftcapping, null,
                null, null, null, null, List.of("DiffusionGemmaForBlockDiffusion"), true);
        this.textConfig = text(textConfig);
        this.visionConfig = normalizeVisionConfig(visionConfig);
        this.boiTokenId = boiTokenId == null ? 255_999 : boiTokenId;
        this.eoiTokenId = eoiTokenId == null ? 258_882 : eoiTokenId;
        this.imageTokenId = imageTokenId == null ? 258_880 : imageTokenId;
        this.initializerRange = initializerRange == null ? 0.02f : initializerRange;
        this.tieWordEmbeddings = tieWordEmbeddings == null || tieWordEmbeddings;
        this.canvasLength = canvasLength == null ? 256 : canvasLength;
    }

    private static DiffusionGemmaTextConfig text(DiffusionGemmaTextConfig textConfig) {
        return textConfig == null ? new DiffusionGemmaTextConfig() : textConfig;
    }

    @SuppressWarnings("unchecked")
    private static List<Integer> eosTokens(Object eosTokenId) {
        if (eosTokenId instanceof List<?> list) {
            return (List<Integer>) list;
        }
        return List.of((Integer) eosTokenId);
    }

    private static Map<String, Object> normalizeVisionConfig(Map<String, Object> visionConfig) {
        if (visionConfig == null) {
            return null;
        }
        Map<String, Object> normalized = new LinkedHashMap<>(visionConfig);
        normalized.putIfAbsent("model_type", "gemma4_vision");
        return Collections.unmodifiableMap(normalized);
    }
}
