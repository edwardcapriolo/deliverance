package io.teknek.deliverance.model.embeddinggemma2;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.teknek.deliverance.math.ActivationFunction;
import io.teknek.deliverance.safetensors.Config;

import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Collections;

/**
 * Root configuration for EmbeddingGemma 2.
 *
 * <p>This configuration reader intentionally supports the composite model
 * metadata before the vision and audio execution paths exist. The nested
 * modality maps are retained for later phases rather than discarded during
 * deserialization.</p>
 */
public final class EmbeddingGemma2Config extends Config {
    public final String modelType;
    public final List<String> architectures;
    public final EmbeddingGemma2TextConfig textConfig;
    public final Map<String, Object> visionConfig;
    public final Map<String, Object> audioConfig;
    public final int imageTokenId;
    public final int videoTokenId;
    public final int audioTokenId;
    public final int boiTokenId;
    public final int eoiTokenId;
    public final int boaTokenId;
    public final int eoaTokenIndex;

    /**
     * Creates the root configuration from Hugging Face {@code config.json}.
     *
     * @param modelType root model type
     * @param architectures model architecture names
     * @param textConfig nested text configuration
     * @param visionConfig nested vision configuration
     * @param audioConfig nested audio configuration
     * @param imageTokenId image placeholder token ID
     * @param videoTokenId video placeholder token ID
     * @param audioTokenId audio placeholder token ID
     * @param boiTokenId begin-image token ID
     * @param eoiTokenId end-image token ID
     * @param boaTokenId begin-audio token ID
     * @param eoaTokenIndex end-audio token ID
     */
    @JsonCreator
    public EmbeddingGemma2Config(
            @JsonProperty("model_type") String modelType,
            @JsonProperty("architectures") List<String> architectures,
            @JsonProperty("text_config") Map<String, Object> textConfig,
            @JsonProperty("vision_config") Map<String, Object> visionConfig,
            @JsonProperty("audio_config") Map<String, Object> audioConfig,
            @JsonProperty("image_token_id") Integer imageTokenId,
            @JsonProperty("video_token_id") Integer videoTokenId,
            @JsonProperty("audio_token_id") Integer audioTokenId,
            @JsonProperty("boi_token_id") Integer boiTokenId,
            @JsonProperty("eoi_token_id") Integer eoiTokenId,
            @JsonProperty("boa_token_id") Integer boaTokenId,
            @JsonProperty("eoa_token_index") Integer eoaTokenIndex
    ) {
        this(
                modelType,
                architectures,
                new EmbeddingGemma2TextConfig(textConfig),
                visionConfig,
                audioConfig,
                imageTokenId,
                videoTokenId,
                audioTokenId,
                boiTokenId,
                eoiTokenId,
                boaTokenId,
                eoaTokenIndex
        );
    }

    private EmbeddingGemma2Config(
            String modelType,
            List<String> architectures,
            EmbeddingGemma2TextConfig textConfig,
            Map<String, Object> visionConfig,
            Map<String, Object> audioConfig,
            Integer imageTokenId,
            Integer videoTokenId,
            Integer audioTokenId,
            Integer boiTokenId,
            Integer eoiTokenId,
            Integer boaTokenId,
            Integer eoaTokenIndex
    ) {
        super(
                textConfig.maxPositionEmbeddings,
                textConfig.embeddingDim,
                textConfig.hiddenSize,
                textConfig.numAttentionHeads,
                textConfig.numKeyValueHeads,
                textConfig.numHiddenLayers,
                textConfig.rmsNormEps,
                textConfig.vocabSize,
                textConfig.bosTokenId,
                textConfig.eosTokenId,
                ActivationFunction.Type.valueOf(textConfig.hiddenActivation.toUpperCase()),
                null,
                null,
                null,
                textConfig.headDim,
                null,
                null,
                null,
                null,
                null,
                null,
                architectures
        );
        this.modelType = requireModelType(modelType);
        this.architectures = architectures == null ? List.of() : List.copyOf(architectures);
        this.textConfig = textConfig;
        this.visionConfig = immutableMap(visionConfig);
        this.audioConfig = immutableMap(audioConfig);
        this.imageTokenId = required(imageTokenId, "image_token_id");
        this.videoTokenId = required(videoTokenId, "video_token_id");
        this.audioTokenId = required(audioTokenId, "audio_token_id");
        this.boiTokenId = required(boiTokenId, "boi_token_id");
        this.eoiTokenId = required(eoiTokenId, "eoi_token_id");
        this.boaTokenId = required(boaTokenId, "boa_token_id");
        this.eoaTokenIndex = required(eoaTokenIndex, "eoa_token_index");
    }

    private static String requireModelType(String modelType) {
        if (!"embedding_gemma2".equals(modelType)) {
            throw new IllegalArgumentException("Expected model_type embedding_gemma2 but found " + modelType);
        }
        return modelType;
    }

    private static int required(Integer value, String name) {
        if (value == null) {
            throw new IllegalArgumentException("Missing " + name + " in EmbeddingGemma 2 config");
        }
        return value;
    }

    private static Map<String, Object> immutableMap(Map<String, Object> value) {
        return value == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(value));
    }
}
