package io.teknek.reckoning.model.tensorparallel;

import com.google.common.base.Preconditions;
import io.teknek.reckoning.model.AbstractModel;
import io.teknek.reckoning.model.tensorparallel.transport.SharedKvPrefixRestoreRequest;
import io.teknek.reckoning.model.tensorparallel.transport.SharedKvPrefixRestoreResult;
import io.teknek.reckoning.model.tensorparallel.transport.SharedKvPrefixStoreRequest;
import io.teknek.reckoning.model.tensorparallel.transport.TensorParallelRankService;
import io.teknek.reckoning.tensor.kv.KvCacheSession;
import io.teknek.reckoning.tensor2.TensorRef;

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
    public synchronized TensorRef batchForward(UUID sessionId, int[] tokenIds, int startPosition) {
        KvCacheSession kvSession = kvSession(sessionId);
        try (var ignored = model.getTensorParallelCollectives().enterSession(sessionId)) {
            return model.batchForwardRef(tokenIds, startPosition, kvSession);
        }
    }

    @Override
    public synchronized TensorRef forward(UUID sessionId, int tokenId, int position) {
        KvCacheSession kvSession = kvSession(sessionId);
        try (var ignored = model.getTensorParallelCollectives().enterSession(sessionId)) {
            return model.forwardRef(tokenId, position, kvSession, Optional.empty());
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
