package io.teknek.deliverance.model;

import io.teknek.deliverance.DType;
import io.teknek.deliverance.JsonUtils;
import io.teknek.deliverance.tensor.KvBufferCacheSettings;
import io.teknek.deliverance.tensorlib.TensorRuntimeMode;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
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
        Optional<Boolean> gpuPrefill,
        Optional<Boolean> gpuDecode,
        Optional<Boolean> gpuDecodeAttention,
        Optional<Boolean> gpuDiffusionBlockProjection,
        Optional<Boolean> packedBlockAttention,
        Optional<Boolean> packedPrefill,
        Optional<Boolean> download,
        Optional<Integer> maxBatchSize,
        Optional<TensorRuntimeMode> tensorRuntimeMode,
        Optional<Map<TensorProviderKind, Integer>> parallelSplitSizeFixed,
        Optional<Map<TensorProviderKind, Double>> parallelSplitSizeMultiplier,
        Optional<Integer> groupedDecodeQkvSplitSize,
        Optional<Boolean> tensorPlanTrace,
        Optional<Map<String, Object>> generationOptions,
        Optional<KvBufferCache> kvBufferCache,
        Optional<QuantizeOnDemand> quantizeOnDemand) {

    public AutoModelConfig {
        workingMemoryType = workingMemoryType == null ? Optional.empty() : workingMemoryType;
        workingQuantType = workingQuantType == null ? Optional.empty() : workingQuantType;
        outputHeadQuantization = outputHeadQuantization == null ? Optional.empty() : outputHeadQuantization;
        gpuPrefill = gpuPrefill == null ? Optional.empty() : gpuPrefill;
        gpuDecode = gpuDecode == null ? Optional.empty() : gpuDecode;
        gpuDecodeAttention = gpuDecodeAttention == null ? Optional.empty() : gpuDecodeAttention;
        gpuDiffusionBlockProjection = gpuDiffusionBlockProjection == null ? Optional.empty() : gpuDiffusionBlockProjection;
        packedBlockAttention = packedBlockAttention == null ? Optional.empty() : packedBlockAttention;
        packedPrefill = packedPrefill == null ? Optional.empty() : packedPrefill;
        download = download == null ? Optional.empty() : download;
        maxBatchSize = maxBatchSize == null ? Optional.empty() : maxBatchSize;
        tensorRuntimeMode = tensorRuntimeMode == null ? Optional.empty() : tensorRuntimeMode;
        parallelSplitSizeFixed = parallelSplitSizeFixed == null ? Optional.empty() : parallelSplitSizeFixed;
        parallelSplitSizeMultiplier = parallelSplitSizeMultiplier == null ? Optional.empty() : parallelSplitSizeMultiplier;
        groupedDecodeQkvSplitSize = groupedDecodeQkvSplitSize == null ? Optional.empty() : groupedDecodeQkvSplitSize;
        tensorPlanTrace = tensorPlanTrace == null ? Optional.empty() : tensorPlanTrace;
        generationOptions = generationOptions == null ? Optional.empty() : generationOptions;
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
            Optional<KvBufferCacheSettings.PrefixCheckpointPolicy> prefixCheckpointPolicy,
            Optional<KvBufferCacheSettings.PrefixCompression> prefixCompression,
            Optional<Integer> prefixTurboQuantBits,
            Optional<Integer> maxPrefixCheckpointsPerPrompt,
            Optional<List<Integer>> prefixCheckpointAnchors,
            Optional<KvBufferCacheSettings.PrefixCacheMode> prefixCacheMode,
            Optional<Long> sharedPrefixBlockCacheMaxBytes,
            Optional<Boolean> sharedPrefixDiskCacheEnabled,
            Optional<File> sharedPrefixDiskCachePath,
            Optional<Long> sharedPrefixDiskCacheMaxBytes,
            Optional<Long> sharedPrefixDiskCacheReservedFreeBytes,
            Optional<Long> sharedPrefixDiskCacheMinUsableBytes,
            Optional<Integer> sharedPrefixDiskCacheAdmitMinTokens,
            Optional<Integer> sharedPrefixDiskCacheWriterQueueSize,
            Optional<Integer> contextRowsPerPageTarget,
            Optional<KvBufferCacheSettings.KvBlockStoragePolicy> kvBlockStoragePolicy,
            Optional<Integer> kvTurboQuantBits,
            Optional<DType> kvKeyDType,
            Optional<DType> kvValueDType) {

        public KvBufferCache {
            maxEntries = maxEntries == null ? Optional.empty() : maxEntries;
            blockSize = blockSize == null ? Optional.empty() : blockSize;
            maxPrefixTokensPerPrompt = maxPrefixTokensPerPrompt == null ? Optional.empty() : maxPrefixTokensPerPrompt;
            prefixCheckpointPolicy = prefixCheckpointPolicy == null ? Optional.empty() : prefixCheckpointPolicy;
            prefixCompression = prefixCompression == null ? Optional.empty() : prefixCompression;
            prefixTurboQuantBits = prefixTurboQuantBits == null ? Optional.empty() : prefixTurboQuantBits;
            maxPrefixCheckpointsPerPrompt = maxPrefixCheckpointsPerPrompt == null ? Optional.empty() : maxPrefixCheckpointsPerPrompt;
            prefixCheckpointAnchors = prefixCheckpointAnchors == null ? Optional.empty() : prefixCheckpointAnchors;
            prefixCacheMode = prefixCacheMode == null ? Optional.empty() : prefixCacheMode;
            sharedPrefixBlockCacheMaxBytes = sharedPrefixBlockCacheMaxBytes == null ? Optional.empty() : sharedPrefixBlockCacheMaxBytes;
            sharedPrefixDiskCacheEnabled = sharedPrefixDiskCacheEnabled == null ? Optional.empty() : sharedPrefixDiskCacheEnabled;
            sharedPrefixDiskCachePath = sharedPrefixDiskCachePath == null ? Optional.empty() : sharedPrefixDiskCachePath;
            sharedPrefixDiskCacheMaxBytes = sharedPrefixDiskCacheMaxBytes == null ? Optional.empty() : sharedPrefixDiskCacheMaxBytes;
            sharedPrefixDiskCacheReservedFreeBytes = sharedPrefixDiskCacheReservedFreeBytes == null ? Optional.empty() : sharedPrefixDiskCacheReservedFreeBytes;
            sharedPrefixDiskCacheMinUsableBytes = sharedPrefixDiskCacheMinUsableBytes == null ? Optional.empty() : sharedPrefixDiskCacheMinUsableBytes;
            sharedPrefixDiskCacheAdmitMinTokens = sharedPrefixDiskCacheAdmitMinTokens == null ? Optional.empty() : sharedPrefixDiskCacheAdmitMinTokens;
            sharedPrefixDiskCacheWriterQueueSize = sharedPrefixDiskCacheWriterQueueSize == null ? Optional.empty() : sharedPrefixDiskCacheWriterQueueSize;
            contextRowsPerPageTarget = contextRowsPerPageTarget == null ? Optional.empty() : contextRowsPerPageTarget;
            kvBlockStoragePolicy = kvBlockStoragePolicy == null ? Optional.empty() : kvBlockStoragePolicy;
            kvTurboQuantBits = kvTurboQuantBits == null ? Optional.empty() : kvTurboQuantBits;
            kvKeyDType = kvKeyDType == null ? Optional.empty() : kvKeyDType;
            kvValueDType = kvValueDType == null ? Optional.empty() : kvValueDType;
        }

        KvBufferCacheSettings toSettings() {
            KvBufferCacheSettings settings = new KvBufferCacheSettings(true);
            maxEntries.ifPresent(settings::setMaxEntries);
            blockSize.ifPresent(settings::setBlockSize);
            maxPrefixTokensPerPrompt.ifPresent(settings::setMaxPrefixTokensPerPrompt);
            prefixCheckpointPolicy.ifPresent(settings::setPrefixCheckpointPolicy);
            prefixCompression.ifPresent(settings::setPrefixCompression);
            prefixTurboQuantBits.ifPresent(settings::setPrefixTurboQuantBits);
            maxPrefixCheckpointsPerPrompt.ifPresent(settings::setMaxPrefixCheckpointsPerPrompt);
            prefixCheckpointAnchors.ifPresent(settings::setPrefixCheckpointAnchors);
            prefixCacheMode.ifPresent(settings::setPrefixCacheMode);
            sharedPrefixBlockCacheMaxBytes.ifPresent(settings::setSharedPrefixBlockCacheMaxBytes);
            sharedPrefixDiskCacheEnabled.ifPresent(settings::setSharedPrefixDiskCacheEnabled);
            sharedPrefixDiskCachePath.ifPresent(settings::setSharedPrefixDiskCachePath);
            sharedPrefixDiskCacheMaxBytes.ifPresent(settings::setSharedPrefixDiskCacheMaxBytes);
            sharedPrefixDiskCacheReservedFreeBytes.ifPresent(settings::setSharedPrefixDiskCacheReservedFreeBytes);
            sharedPrefixDiskCacheMinUsableBytes.ifPresent(settings::setSharedPrefixDiskCacheMinUsableBytes);
            sharedPrefixDiskCacheAdmitMinTokens.ifPresent(settings::setSharedPrefixDiskCacheAdmitMinTokens);
            sharedPrefixDiskCacheWriterQueueSize.ifPresent(settings::setSharedPrefixDiskCacheWriterQueueSize);
            contextRowsPerPageTarget.ifPresent(settings::setContextRowsPerPageTarget);
            kvBlockStoragePolicy.ifPresent(settings::setKvBlockStoragePolicy);
            kvTurboQuantBits.ifPresent(settings::setKvTurboQuantBits);
            kvKeyDType.ifPresent(settings::setKvKeyDType);
            kvValueDType.ifPresent(settings::setKvValueDType);
            return settings;
        }
    }

    public record QuantizeOnDemand(DType targetType, String outputOwner, String outputModel) {
    }
}
