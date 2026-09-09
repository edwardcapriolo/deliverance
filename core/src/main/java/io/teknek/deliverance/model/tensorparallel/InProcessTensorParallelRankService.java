package io.teknek.deliverance.model.tensorparallel;

import com.google.common.base.Preconditions;
import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.tensorparallel.transport.SharedKvPrefixRestoreRequest;
import io.teknek.deliverance.model.tensorparallel.transport.SharedKvPrefixRestoreResult;
import io.teknek.deliverance.model.tensorparallel.transport.SharedKvPrefixStoreRequest;
import io.teknek.deliverance.model.tensorparallel.transport.TensorParallelRankService;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.kv.KvCacheSession;

import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Rank service adapter for a local in-process model instance.
 */
public class InProcessTensorParallelRankService implements TensorParallelRankService, AutoCloseable {
    private final AbstractModel model;
    private final Map<UUID, KvCacheSession> kvSessions = new ConcurrentHashMap<>();

    public InProcessTensorParallelRankService(AbstractModel model) {
        Preconditions.checkArgument(model.usesKvCache2Generation(), "Tensor parallel generation requires KVCache2");
        this.model = model;
    }

    @Override
    public synchronized AbstractTensor batchForward(UUID sessionId, int[] tokenIds, int startPosition) {
        KvCacheSession kvSession = kvSession(sessionId);
        try (var ignored = model.getTensorParallelCollectives().enterSession(sessionId)) {
            return model.batchForward(tokenIds, startPosition, kvSession);
        }
    }

    @Override
    public synchronized AbstractTensor forward(UUID sessionId, int tokenId, int position) {
        KvCacheSession kvSession = kvSession(sessionId);
        try (var ignored = model.getTensorParallelCollectives().enterSession(sessionId)) {
            return model.forward(tokenId, position, kvSession, Optional.empty());
        }
    }

    @Override
    public synchronized SharedKvPrefixRestoreResult restoreSharedKvPrefix(SharedKvPrefixRestoreRequest request) {
        KvCacheSession kvSession = kvSession(request.sessionId());
        int prefixLength = model.restoreSharedPrefixToKvSession(request.tokenIds(), Optional.ofNullable(request.cacheSalt()),
                kvSession);
        return new SharedKvPrefixRestoreResult(prefixLength);
    }

    @Override
    public synchronized void storeSharedKvPrefix(SharedKvPrefixStoreRequest request) {
        KvCacheSession kvSession = kvSessions.get(request.sessionId());
        if (kvSession != null) {
            model.storeSharedPrefixFromKvSession(request.tokenIds(), kvSession, Optional.ofNullable(request.cacheSalt()));
        }
    }

    public void closeSession(UUID sessionId) {
        KvCacheSession kvSession = kvSessions.remove(sessionId);
        if (kvSession != null) {
            kvSession.close();
        }
        model.getTensorParallelCollectives().closeSession(sessionId);
    }

    @Override
    public void close() {
        for (KvCacheSession kvSession : kvSessions.values()) {
            kvSession.close();
        }
        kvSessions.clear();
        model.close();
    }

    private KvCacheSession kvSession(UUID sessionId) {
        return kvSessions.computeIfAbsent(sessionId, ignored -> model.newKvCacheSession());
    }
}
