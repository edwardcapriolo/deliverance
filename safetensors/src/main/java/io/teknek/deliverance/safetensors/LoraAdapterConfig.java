package io.teknek.deliverance.safetensors;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.List;
import java.util.Objects;

/**
 * Parsed view of a standard HuggingFace PEFT {@code adapter_config.json}.
 *
 * <p>Only the fields Deliverance's LoRA support needs are modeled here. Real-world
 * {@code adapter_config.json} files carry many additional fields (e.g. {@code
 * base_model_name_or_path}, {@code task_type}, {@code bias}, {@code lora_dropout}, {@code
 * peft_type}) that are intentionally ignored rather than rejected.</p>
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class LoraAdapterConfig {

    public static final String FILE_NAME = "adapter_config.json";

    @JsonProperty("r")
    public final int rank;

    @JsonProperty("lora_alpha")
    public final double alpha;

    @JsonProperty("target_modules")
    public final List<String> targetModules;

    @JsonCreator
    public LoraAdapterConfig(
            @JsonProperty("r") int rank,
            @JsonProperty("lora_alpha") double alpha,
            @JsonProperty("target_modules") List<String> targetModules) {
        if (rank <= 0) {
            throw new IllegalArgumentException("LoRA adapter rank (r) must be positive, got " + rank);
        }
        if (targetModules == null || targetModules.isEmpty()) {
            throw new IllegalArgumentException("LoRA adapter config has no target_modules");
        }
        this.rank = rank;
        this.alpha = alpha;
        this.targetModules = List.copyOf(targetModules);
    }

    /** The standard LoRA merge scale, {@code alpha / r}. */
    public double scale() {
        return alpha / rank;
    }

    public static LoraAdapterConfig load(File adapterDir) {
        File configFile = new File(adapterDir, FILE_NAME);
        try {
            return JsonUtils.om.readValue(configFile, LoraAdapterConfig.class);
        } catch (IOException e) {
            throw new UncheckedIOException("Unable to read " + configFile, e);
        }
    }

    @Override
    public String toString() {
        return "LoraAdapterConfig{" + "rank=" + rank + ", alpha=" + alpha + ", targetModules=" + targetModules + '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof LoraAdapterConfig that)) return false;
        return rank == that.rank && Double.compare(alpha, that.alpha) == 0 && targetModules.equals(that.targetModules);
    }

    @Override
    public int hashCode() {
        return Objects.hash(rank, alpha, targetModules);
    }
}
