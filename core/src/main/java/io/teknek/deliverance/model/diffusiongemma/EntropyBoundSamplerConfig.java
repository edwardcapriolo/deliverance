package io.teknek.deliverance.model.diffusiongemma;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/** Configuration for {@link EntropyBoundSampler}. */
@JsonIgnoreProperties(ignoreUnknown = true)
public final class EntropyBoundSamplerConfig {
    @JsonProperty("entropy_bound")
    public final float entropyBound;

    @JsonCreator
    public EntropyBoundSamplerConfig(@JsonProperty("entropy_bound") Float entropyBound) {
        if (entropyBound == null || !Float.isFinite(entropyBound) || entropyBound <= 0.0f) {
            throw new IllegalArgumentException("entropyBound must be finite and > 0");
        }
        this.entropyBound = entropyBound;
    }
}
