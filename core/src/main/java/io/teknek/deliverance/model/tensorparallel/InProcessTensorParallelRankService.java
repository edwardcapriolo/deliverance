package io.teknek.deliverance.model.tensorparallel;

import io.teknek.deliverance.model.AbstractModel;
import io.teknek.deliverance.model.tensorparallel.transport.TensorParallelRankService;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;

import java.util.Map;
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
    public AbstractTensor batchForward(UUID sessionId, int[] tokenIds, int startPosition) {
        return model.batchForward(tokenIds, startPosition, kvBuffer(sessionId));
    }

    @Override
    public AbstractTensor forward(UUID sessionId, int tokenId, int position) {
        return model.forward(tokenId, position, kvBuffer(sessionId), java.util.Optional.empty());
    }

    public void closeSession(UUID sessionId) {
        KvBufferCache.KvBuffer kvBuffer = kvBuffers.remove(sessionId);
        if (kvBuffer != null) {
            kvBuffer.close();
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
}
