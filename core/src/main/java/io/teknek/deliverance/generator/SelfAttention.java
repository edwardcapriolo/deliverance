package io.teknek.deliverance.generator;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;

import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

public interface SelfAttention {
    AbstractTensor forward(AbstractTensor input, int startPosition, KvBufferCache.KvBuffer kvMem,
            Optional<Consumer<List<AbstractTensor>>> tensorReducer);
}
