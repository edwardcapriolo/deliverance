package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.tensor.AbstractTensor;

import java.util.UUID;

/**
 * Internal execution boundary used by {@link GenerationEngine}.
 *
 * <p>Implementations own the KV/session mechanics for a request. For example, the local backend owns a local
 * {@code KvBuffer}, while tensor-parallel backends can dispatch prefill and decode to rank services.</p>
 */
public interface GenerationBackend extends AutoCloseable {
    /**
     * Opens execution state for one generation request.
     *
     * @param sessionId caller-provided request/session id; distributed backends use it to identify rank-local KV state
     * @param promptTokens full prompt tokens including any BOS token inserted by the coordinator model
     * @param parameters generation parameters for this request; backends may use cache-related fields
     * @return a session that must be closed by the caller after generation completes or fails
     */
    GenerationSession open(UUID sessionId, int[] promptTokens, GeneratorParameters parameters);

    @Override
    default void close() {
    }

    /**
     * Per-request execution state opened by a {@link GenerationBackend}.
     *
     * <p>A session lasts for one generation request and owns any temporary KV state, remote session id, or cleanup work
     * needed by that backend.</p>
     */
    interface GenerationSession extends AutoCloseable {
        /**
         * Returns how many prompt tokens are already represented in backend KV state.
         *
         * <p>Local prefix-cache hits return a positive block-aligned length. Tensor-parallel and other remote backends may
         * return zero even when a semantically identical local request would hit the prefix cache, because their KV state is
         * managed behind the backend boundary. The generation engine uses this value to build the {@link GenerationCursor};
         * implementations must never return a value outside {@code [0, promptTokens.length]}.</p>
         */
        int prefixLength();

        /**
         * Runs prompt prefill for the uncached part of the prompt and returns the final hidden state used for first-token
         * sampling.
         *
         * <p>The returned tensor is owned by the caller and must be closed after sampling. Implementations must leave KV
         * state positioned so subsequent {@link #decode(int, int)} calls continue at the requested decode positions.</p>
         */
        AbstractTensor prefill(GenerationCursor cursor);

        /**
         * Runs one decode step for a previously sampled token.
         *
         * @param tokenId token sampled by the generation engine on the previous step
         * @param position absolute token position in the sequence
         * @return hidden state for this decode step; caller owns and closes the returned tensor
         */
        AbstractTensor decode(int tokenId, int position);

        /**
         * Notifies the backend that a decode step completed successfully.
         *
         * <p>Local KV buffers use this hook to advance their current context position. Backends that manage position
         * externally can leave the default no-op implementation.</p>
         */
        default void afterDecode() {
        }

        @Override
        void close();
    }
}
