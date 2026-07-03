package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;

import java.util.UUID;

/**
 * Local generation backend backed by one in-process transformer executor.
 *
 * <p>This backend owns the ephemeral local KV buffer for a generation request. It is also responsible for local prefix
 * cache lookup, prefix copy, prefix storage after prompt prefill, and advancing the KV context position after decode
 * steps. The shared {@link GenerationEngine} handles token sampling and stop conditions.</p>
 */
public final class LocalGenerationBackend implements GenerationBackend {
    private final AbstractModel model;

    public LocalGenerationBackend(AbstractModel model) {
        this.model = java.util.Objects.requireNonNull(model, "model");
    }

    /**
     * Opens a local generation session and performs prefix-cache lookup for the full prompt.
     *
     * <p>The session owns one ephemeral KV buffer. If a prefix-cache hit is found, the cached KV rows are copied before
     * prefill runs, and {@link GenerationSession#prefixLength()} reports the copied prefix length.</p>
     */
    @Override
    public GenerationSession open(UUID sessionId, int[] promptTokens, GeneratorParameters parameters) {
        return new LocalGenerationSession(promptTokens, parameters);
    }

    /** Per-request local KV state for {@link LocalGenerationBackend}. */
    private final class LocalGenerationSession implements GenerationSession {
        private final int[] promptTokens;
        private final GeneratorParameters parameters;
        private final KvBufferCache.KvBuffer kvBuffer;
        private final int prefixLength;

        private LocalGenerationSession(int[] promptTokens, GeneratorParameters parameters) {
            this.promptTokens = promptTokens;
            this.parameters = parameters;
            this.kvBuffer = model.kvBufferCache.getEphemeralKvBuffer();
            KvBufferCache.PrefixEntry prefixHit = model.kvBufferCache.lookupPrefix(promptTokens, parameters.cacheSalt);
            if (prefixHit == null) {
                this.prefixLength = 0;
            } else {
                this.prefixLength = prefixHit.length();
                try {
                    model.kvBufferCache.copyPrefix(prefixHit.buffer(), kvBuffer, prefixLength);
                } finally {
                    prefixHit.closeIfTemporary();
                }
                model.emitGenerationDebug(new AbstractModel.GenerationDebugEvent(
                        AbstractModel.GenerationDebugEventType.AFTER_PREFIX_COPY,
                        promptTokens,
                        prefixLength,
                        prefixLength,
                        promptTokens.length - prefixLength,
                        kvBuffer));
            }
        }

        /** Returns the prefix-cache hit length copied into this session's KV buffer, or zero on a miss. */
        @Override
        public int prefixLength() {
            return prefixLength;
        }

        /**
         * Runs local prompt prefill from the cursor start position.
         *
         * <p>If the entire prompt was restored from the prefix cache, this replays the last cached token to produce the
         * hidden state required for first-token sampling without changing the prompt length contract. Otherwise it runs
         * {@link AbstractModel#batchForward(int[], int, KvBufferCache.KvBuffer)} for the uncached suffix and stores the
         * full prompt in the prefix cache after prefill.</p>
         */
        @Override
        public AbstractTensor prefill(GenerationCursor cursor) {
            kvBuffer.setCurrentContextPosition(cursor.startPosition());
            AbstractTensor last;
            if (cursor.hasTokensToProcess()) {
                last = model.batchForward(cursor.tokensToProcess(), cursor.startPosition(), kvBuffer);
                model.kvBufferCache.storePrefix(promptTokens, kvBuffer, parameters.cacheSalt);
            } else {
                last = model.forward(cursor.replayToken(), cursor.replayPosition(), kvBuffer);
            }
            model.emitGenerationDebug(new AbstractModel.GenerationDebugEvent(
                    AbstractModel.GenerationDebugEventType.AFTER_PROMPT_PREFILL,
                    promptTokens,
                    prefixLength,
                    cursor.startPosition(),
                    cursor.tokensToProcess().length,
                    kvBuffer));
            return last;
        }

        /** Runs one local decode step against this session's KV buffer. */
        @Override
        public AbstractTensor decode(int tokenId, int position) {
            return model.forward(tokenId, position, kvBuffer);
        }

        /** Advances the local KV cursor after a successful decode step. */
        @Override
        public void afterDecode() {
            kvBuffer.incrementContextPosition();
        }

        /** Releases the ephemeral KV buffer owned by this request. */
        @Override
        public void close() {
            kvBuffer.close();
        }
    }
}
