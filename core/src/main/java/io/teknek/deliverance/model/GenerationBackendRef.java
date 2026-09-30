package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.UUID;

/** TensorRef-native generation boundary for KV-cache v2 execution. */
public interface GenerationBackendRef extends AutoCloseable {
    GenerationSessionRef openRef(UUID sessionId, int[] promptTokens, GeneratorParameters parameters);

    @Override
    default void close() {
    }

    interface GenerationSessionRef extends AutoCloseable {
        int prefixLength();
        TensorRef prefill(GenerationCursor cursor);
        TensorRef decode(int tokenId, int position);

        default void afterDecode() {
        }

        @Override
        void close();
    }
}
