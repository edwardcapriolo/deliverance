package io.teknek.deliverance.model.diffusiongemma;

import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Generation-time configuration for DiffusionGemma's block diffusion loop.
 *
 * <p>This intentionally accepts only the autoregressive generation fields that HF DiffusionGemma allows plus the
 * diffusion-specific denoising controls. Common AR sampling parameters such as top-k, top-p, beam search, and repetition
 * penalties are rejected because DiffusionGemma uses canvas denoising, entropy-bound acceptance, and renoising instead of
 * normal next-token sampling.</p>
 */
public final class DiffusionGemmaGenerationConfig {
    private static final Set<String> REJECTED_AR_FIELDS = Set.of(
            "do_sample",
            "num_beams",
            "num_beam_groups",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "encoder_no_repeat_ngram_size",
            "length_penalty",
            "early_stopping",
            "num_return_sequences",
            "foo");
    private static final Set<String> HF_METADATA_FIELDS = Set.of(
            "transformers_version",
            "_commit_hash",
            "_from_model_config");

    @JsonProperty("max_length")
    public final Integer maxLength;
    @JsonProperty("max_new_tokens")
    public final Integer maxNewTokens;
    @JsonProperty("max_denoising_steps")
    public final Integer maxDenoisingSteps;
    @JsonProperty("sampler_config")
    public final EntropyBoundSamplerConfig samplerConfig;
    @JsonProperty("t_min")
    public final Float tMin;
    @JsonProperty("t_max")
    public final Float tMax;
    @JsonProperty("stability_threshold")
    public final Integer stabilityThreshold;
    @JsonProperty("confidence_threshold")
    public final Float confidenceThreshold;
    @JsonProperty("cache_implementation")
    public final String cacheImplementation;
    @JsonProperty("cache_config")
    public final Map<String, Object> cacheConfig;
    @JsonProperty("bos_token_id")
    public final Integer bosTokenId;
    @JsonProperty("pad_token_id")
    public final Integer padTokenId;
    @JsonProperty("eos_token_id")
    public final Object eosTokenId;

    public DiffusionGemmaGenerationConfig() {
        this(256, null, 48, new EntropyBoundSamplerConfig(0.1f), 0.4f, 0.8f, 1,
                0.005f, null, null, null, null, null);
    }

    @JsonCreator
    public DiffusionGemmaGenerationConfig(
            @JsonProperty("max_length") Integer maxLength,
            @JsonProperty("max_new_tokens") Integer maxNewTokens,
            @JsonProperty("max_denoising_steps") Integer maxDenoisingSteps,
            @JsonProperty("sampler_config") EntropyBoundSamplerConfig samplerConfig,
            @JsonProperty("t_min") Float tMin,
            @JsonProperty("t_max") Float tMax,
            @JsonProperty("stability_threshold") Integer stabilityThreshold,
            @JsonProperty("confidence_threshold") Float confidenceThreshold,
            @JsonProperty("cache_implementation") String cacheImplementation,
            @JsonProperty("cache_config") Map<String, Object> cacheConfig,
            @JsonProperty("bos_token_id") Integer bosTokenId,
            @JsonProperty("pad_token_id") Integer padTokenId,
            @JsonProperty("eos_token_id") Object eosTokenId) {
        this.maxLength = maxLength;
        this.maxNewTokens = maxNewTokens;
        this.maxDenoisingSteps = maxDenoisingSteps;
        this.samplerConfig = samplerConfig;
        this.tMin = tMin;
        this.tMax = tMax;
        this.stabilityThreshold = stabilityThreshold;
        this.confidenceThreshold = confidenceThreshold;
        this.cacheImplementation = cacheImplementation;
        this.cacheConfig = cacheConfig;
        this.bosTokenId = bosTokenId;
        this.padTokenId = padTokenId;
        this.eosTokenId = eosTokenId;
        validate();
    }

    @JsonAnySetter
    void rejectUnknownField(String name, Object value) {
        if (HF_METADATA_FIELDS.contains(name)) {
            return;
        }
        if (REJECTED_AR_FIELDS.contains(name)) {
            throw new IllegalArgumentException("DiffusionGemmaGenerationConfig does not support `" + name + "`");
        }
        throw new IllegalArgumentException("Unexpected DiffusionGemmaGenerationConfig field `" + name + "`");
    }

    private void validate() {
        if (maxLength != null && maxLength <= 0) {
            throw new IllegalArgumentException("maxLength must be > 0");
        }
        if (maxNewTokens != null && maxNewTokens <= 0) {
            throw new IllegalArgumentException("maxNewTokens must be > 0");
        }
        if (maxDenoisingSteps != null && maxDenoisingSteps <= 0) {
            throw new IllegalArgumentException("maxDenoisingSteps must be > 0");
        }
        if (tMin != null && (!Float.isFinite(tMin) || tMin < 0.0f)) {
            throw new IllegalArgumentException("tMin must be finite and >= 0");
        }
        if (tMax != null && (!Float.isFinite(tMax) || tMax < 0.0f)) {
            throw new IllegalArgumentException("tMax must be finite and >= 0");
        }
        if (tMin != null && tMax != null && tMax <= tMin) {
            throw new IllegalArgumentException("tMax must be > tMin");
        }
        if (stabilityThreshold != null && stabilityThreshold < 0) {
            throw new IllegalArgumentException("stabilityThreshold must be >= 0");
        }
        if (confidenceThreshold != null && (!Float.isFinite(confidenceThreshold) || confidenceThreshold <= 0.0f)) {
            throw new IllegalArgumentException("confidenceThreshold must be finite and > 0");
        }
        if (eosTokenId != null && !(eosTokenId instanceof Integer) && !(eosTokenId instanceof List<?>)) {
            throw new IllegalArgumentException("eosTokenId must be an integer or list of integers");
        }
    }
}
