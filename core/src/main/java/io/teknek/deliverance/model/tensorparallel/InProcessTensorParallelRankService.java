package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.tensorparallel.transport.PrefixCacheProbeRequest;
import io.teknek.deliverance.model.tensorparallel.transport.PrefixCacheProbeResult;
import io.teknek.deliverance.model.tensorparallel.transport.PrefixCacheRestoreRequest;
import io.teknek.deliverance.model.tensorparallel.transport.PrefixCacheRestoreResult;
import io.teknek.deliverance.model.tensorparallel.transport.PrefixCacheStoreRequest;
import io.teknek.deliverance.model.tensorparallel.transport.TensorParallelRankService;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;

import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Rank service adapter for a local in-process model instance.
 */
public class InProcessTensorParallelRankService implements TensorParallelRankService, AutoCloseable {
    private final AbstractModel model;
    private final Map<UUID, KvBufferCache.KvBuffer> kvBuffers = new ConcurrentHashMap<>();

    public InProcessTensorParallelRankService(AbstractModel model) {
        this.model = model;
    }

    @Override
    public synchronized AbstractTensor batchForward(UUID sessionId, int[] tokenIds, int startPosition) {
        KvBufferCache.KvBuffer buffer = kvBuffer(sessionId);
        buffer.setCurrentContextPosition(startPosition);
        try (var ignored = model.getTensorParallelCollectives().enterSession(sessionId)) {
            return model.batchForward(tokenIds, startPosition, buffer);
        }
    }

    @Override
    public synchronized AbstractTensor forward(UUID sessionId, int tokenId, int position) {
        KvBufferCache.KvBuffer buffer = kvBuffer(sessionId);
        try (var ignored = model.getTensorParallelCollectives().enterSession(sessionId)) {
            return model.forward(tokenId, position, buffer, java.util.Optional.empty());
        } finally {
            buffer.incrementContextPosition();
        }
    }

    public void closeSession(UUID sessionId) {
        KvBufferCache.KvBuffer kvBuffer = kvBuffers.remove(sessionId);
        if (kvBuffer != null) {
            kvBuffer.close();
        }
        model.getTensorParallelCollectives().closeSession(sessionId);
    }

    @Override
    public PrefixCacheProbeResult probePrefix(PrefixCacheProbeRequest request) {
        try (KvBufferCache.KvBuffer temporary = model.newKvBuffer()) {
            int prefixLength = model.restorePrefixToKvBuffer(request.tokenIds(), rankCacheSalt(request.cacheSalt()), temporary);
            return new PrefixCacheProbeResult(prefixLength > 0, prefixLength);
        }
    }

    @Override
    public PrefixCacheRestoreResult restorePrefix(PrefixCacheRestoreRequest request) {
        KvBufferCache.KvBuffer buffer = kvBuffer(request.sessionId());
        int prefixLength = model.restorePrefixToKvBuffer(request.tokenIds(), rankCacheSalt(request.cacheSalt()), buffer);
        boolean restored = prefixLength >= request.prefixLength() && request.prefixLength() > 0;
        if (restored && prefixLength != request.prefixLength()) {
            buffer.setCurrentContextPosition(request.prefixLength());
        }
        return new PrefixCacheRestoreResult(restored, restored ? request.prefixLength() : prefixLength);
    }

    @Override
    public void storePrefix(PrefixCacheStoreRequest request) {
        KvBufferCache.KvBuffer buffer = kvBuffers.get(request.sessionId());
        if (buffer != null) {
            model.storePrefixFromKvBuffer(request.tokenIds(), buffer, rankCacheSalt(request.cacheSalt()));
        }
    }

    @Override
    public void close() {
        for (KvBufferCache.KvBuffer kvBuffer : kvBuffers.values()) {
            kvBuffer.close();
        }
        kvBuffers.clear();
        model.close();
    }

    private KvBufferCache.KvBuffer kvBuffer(UUID sessionId) {
        return kvBuffers.computeIfAbsent(sessionId, ignored -> model.newKvBuffer());
    }

    private Optional<String> rankCacheSalt(String requestSalt) {
        String rankSalt = "tp:size=" + model.getTensorParallelContext().size()
                + ":rank=" + model.getTensorParallelContext().rank();
        if (requestSalt == null || requestSalt.isBlank()) {
            return Optional.of(rankSalt);
        }
        return Optional.of(rankSalt + "|" + requestSalt);
    }
}
