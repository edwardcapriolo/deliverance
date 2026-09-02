package io.teknek.deliverance.tensor.kv;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.DType;

import java.util.Objects;

/** Rank-aware identity for one immutable KVCache2 token block. */
public record KvBlockKey(
        int formatVersion,
        String modelCacheId,
        String adapterFingerprint,
        String tokenizerFingerprint,
        String runtimeSalt,
        String ropeConfigHash,
        String attentionConfigHash,
        int blockIndex,
        long parentTokenHash,
        long tokenBlockHash,
        int blockSize,
        int tokenCount,
        int layers,
        int kvLength,
        DType keyDType,
        DType valueDType,
        KvBlockLayout layout,
        int turboQuantBits,
        int tensorParallelSize,
        int tensorParallelRank,
        long assignmentEpoch,
        String localShardId) {
    public KvBlockKey {
        Preconditions.checkArgument(formatVersion > 0, "formatVersion must be > 0");
        modelCacheId = Objects.requireNonNull(modelCacheId, "modelCacheId");
        adapterFingerprint = Objects.requireNonNull(adapterFingerprint, "adapterFingerprint");
        tokenizerFingerprint = Objects.requireNonNull(tokenizerFingerprint, "tokenizerFingerprint");
        runtimeSalt = Objects.requireNonNull(runtimeSalt, "runtimeSalt");
        ropeConfigHash = Objects.requireNonNull(ropeConfigHash, "ropeConfigHash");
        attentionConfigHash = Objects.requireNonNull(attentionConfigHash, "attentionConfigHash");
        keyDType = Objects.requireNonNull(keyDType, "keyDType");
        valueDType = Objects.requireNonNull(valueDType, "valueDType");
        layout = Objects.requireNonNull(layout, "layout");
        localShardId = Objects.requireNonNull(localShardId, "localShardId");
        Preconditions.checkArgument(blockIndex >= 0, "blockIndex must be >= 0");
        Preconditions.checkArgument(blockSize > 0, "blockSize must be > 0");
        Preconditions.checkArgument(tokenCount > 0 && tokenCount <= blockSize, "tokenCount out of bounds");
        Preconditions.checkArgument(layers > 0, "layers must be > 0");
        Preconditions.checkArgument(kvLength > 0, "kvLength must be > 0");
        Preconditions.checkArgument(turboQuantBits >= 0, "turboQuantBits must be >= 0");
        Preconditions.checkArgument(tensorParallelSize > 0, "tensorParallelSize must be > 0");
        Preconditions.checkArgument(tensorParallelRank >= 0 && tensorParallelRank < tensorParallelSize,
                "tensorParallelRank out of bounds");
    }

    public static KvBlockKey local(String modelCacheId, String runtimeSalt, int blockIndex, long parentTokenHash,
            long tokenBlockHash, int blockSize, int tokenCount, int layers, int kvLength, DType keyDType,
            DType valueDType, KvBlockLayout layout, int turboQuantBits) {
        return new KvBlockKey(1, modelCacheId, "none", "unknown", runtimeSalt, "unknown", "unknown", blockIndex,
                parentTokenHash, tokenBlockHash, blockSize, tokenCount, layers, kvLength, keyDType, valueDType, layout,
                turboQuantBits, 1, 0, 0L, "local");
    }
}
