package io.teknek.deliverance.model.tensorparallel.transport;

import io.teknek.deliverance.tensor.AbstractTensor;

import java.util.UUID;

/**
 * Rank-local forward operations exposed to a tensor-parallel coordinator.
 */
public interface TensorParallelRankService {
    AbstractTensor batchForward(UUID sessionId, int[] tokenIds, int startPosition);

    AbstractTensor forward(UUID sessionId, int tokenId, int position);

    default PrefixCacheProbeResult probePrefix(PrefixCacheProbeRequest request) {
        return new PrefixCacheProbeResult(false, 0);
    }

    default PrefixCacheRestoreResult restorePrefix(PrefixCacheRestoreRequest request) {
        return new PrefixCacheRestoreResult(false, 0);
    }

    default void storePrefix(PrefixCacheStoreRequest request) {
    }

    void closeSession(UUID sessionId);
}
