package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.tensor2.TensorRef;

import java.util.UUID;

/**
 * Adapter from the {@link AbstractModel.GenerationForwarder} callback to {@link GenerationBackendRef}.
 *
 * <p>This keeps tensor-parallel generation on the shared {@link GenerationEngineRef} while the rank coordination code still
 * exposes prefill/decode as a pair of forwarding callbacks. Prefix cache length is always zero because KV state is owned
 * by the forwarded implementation rather than by this adapter.</p>
 */
final class ForwarderGenerationBackend implements GenerationBackendRef {
    private final AbstractModel.GenerationForwarder forwarder;

    ForwarderGenerationBackend(AbstractModel.GenerationForwarder forwarder) {
        this.forwarder = java.util.Objects.requireNonNull(forwarder, "forwarder");
    }

    /**
     * Opens a stateless adapter session over the supplied forwarder.
     *
     * <p>The forwarder owns any actual KV/session state. This adapter always reports a zero prefix length and performs no
     * cleanup beyond allowing the generation engine to use a uniform session lifecycle.</p>
     */
    @Override
    public GenerationSessionRef openRef(UUID sessionId, int[] promptTokens, GeneratorParameters parameters) {
        return new GenerationSessionRef() {
            /** Forwarded sessions do not expose prefix-cache reuse to the engine. */
            @Override
            public int prefixLength() {
                return 0;
            }

            /** Delegates prompt prefill to the forwarder. */
            @Override
            public TensorRef prefill(GenerationCursor cursor) {
                return forwarder.batchForward(cursor.tokensToProcess(), cursor.startPosition());
            }

            /** Delegates one decode step to the forwarder. */
            @Override
            public TensorRef decode(int tokenId, int position) {
                return forwarder.forward(tokenId, position);
            }

            /** No-op because the forwarder owns session cleanup. */
            @Override
            public void close() {
            }
        };
    }
}
