package io.teknek.deliverance.model;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Optional;

/**
 * JSON-serializable configuration for {@link AutoModelForCausaLm.Builder}.
 *
 * <p>This exists so benchmark scripts and local experiments do not need a new CLI flag for every builder knob. The config
 * intentionally mirrors stable builder concepts rather than every internal field.</p>
 */
public record AutoModelConfig(
        Optional<DType> workingMemoryType,
        Optional<DType> workingQuantType,
        Optional<DType> outputHeadQuantization,
        Optional<Boolean> download,
        Optional<KvBufferCache> kvBufferCache,
        Optional<QuantizeOnDemand> quantizeOnDemand) {

    public AutoModelConfig {
        workingMemoryType = workingMemoryType == null ? Optional.empty() : workingMemoryType;
        workingQuantType = workingQuantType == null ? Optional.empty() : workingQuantType;
        outputHeadQuantization = outputHeadQuantization == null ? Optional.empty() : outputHeadQuantization;
        download = download == null ? Optional.empty() : download;
        kvBufferCache = kvBufferCache == null ? Optional.empty() : kvBufferCache;
        quantizeOnDemand = quantizeOnDemand == null ? Optional.empty() : quantizeOnDemand;
    }

    public static AutoModelConfig fromJson(File file) {
        try {
            return JsonUtils.om.readValue(file, AutoModelConfig.class);
        } catch (IOException e) {
            throw new RuntimeException("Unable to read auto model config " + file, e);
        }
    }

    public static AutoModelConfig fromJson(Path path) {
        return fromJson(path.toFile());
    }

    public record KvBufferCache(
            Optional<Integer> maxEntries,
            Optional<Integer> blockSize,
            Optional<Integer> maxPrefixTokensPerPrompt,
            Optional<Integer> contextRowsPerPageTarget) {

        public KvBufferCache {
            maxEntries = maxEntries == null ? Optional.empty() : maxEntries;
            blockSize = blockSize == null ? Optional.empty() : blockSize;
            maxPrefixTokensPerPrompt = maxPrefixTokensPerPrompt == null ? Optional.empty() : maxPrefixTokensPerPrompt;
            contextRowsPerPageTarget = contextRowsPerPageTarget == null ? Optional.empty() : contextRowsPerPageTarget;
        }

        KvBufferCacheSettings toSettings() {
            KvBufferCacheSettings settings = new KvBufferCacheSettings(true);
            maxEntries.ifPresent(settings::setMaxEntries);
            blockSize.ifPresent(settings::setBlockSize);
            maxPrefixTokensPerPrompt.ifPresent(settings::setMaxPrefixTokensPerPrompt);
            contextRowsPerPageTarget.ifPresent(settings::setContextRowsPerPageTarget);
            return settings;
        }
    }

    public record QuantizeOnDemand(DType targetType, String outputOwner, String outputModel) {
    }
}
