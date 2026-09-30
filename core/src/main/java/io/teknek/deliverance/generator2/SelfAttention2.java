package io.teknek.deliverance.generator2;

import io.teknek.deliverance.generator.ForwardPhase;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.kv.KvCacheSession;
import io.teknek.deliverance.tensor2.TensorRef;

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
}
