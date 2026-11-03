package io.teknek.reckoning.generator2;

import io.teknek.reckoning.generator.ForwardPhase;
import io.teknek.reckoning.tensor.KvBufferCache;
import io.teknek.reckoning.tensor.kv.KvCacheSession;
import io.teknek.reckoning.tensor.kv.CacheExecutionMode;
import io.teknek.reckoning.tensor2.TensorRef;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

public interface SelfAttention2 {
    TensorRef forward(TensorRef input, int startPosition, KvBufferCache.KvBuffer kvMem,
            Optional<Consumer<List<TensorRef>>> tensorReducer);

    default TensorRef forward(TensorRef input, int startPosition, KvBufferCache.KvBuffer kvMem,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase) {
        return forward(input, startPosition, kvMem, tensorReducer);
    }

    default TensorRef forward(TensorRef input, int startPosition, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support KVCache2");
    }

    default TensorRef forward(TensorRef input, int startPosition, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase, int batchSize,
            int sequenceLength, int[] attentionMask) {
        return forward(input, startPosition, kvSession, tensorReducer, phase);
    }

    default TensorRef forward(TensorRef input, int startPosition, KvCacheSession kvSession,
            Optional<Consumer<List<TensorRef>>> tensorReducer, ForwardPhase phase,
            CacheExecutionMode mode) {
        return forward(input, startPosition, kvSession, tensorReducer, phase);
    }
}
