package io.teknek.deliverance.model;

import io.teknek.deliverance.generator.GeneratorParameters;
import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor.KvBufferCache;
import io.teknek.deliverance.tensor.kv.KvCacheSession;

import java.util.UUID;

/**
 * Local generation backend backed by one in-process transformer executor.
 *
 * <p>This backend owns the local KV state for a generation request. The shared {@link GenerationEngine} handles token
 * sampling and stop conditions.</p>
 */
public final class LocalGenerationBackend implements GenerationBackend {
    private final AbstractModel model;

    public LocalGenerationBackend(AbstractModel model) {
        this.model = java.util.Objects.requireNonNull(model, "model");
    }

    /**
     * Opens a local generation session. Prefix-cache reuse is intentionally disabled; KV session state is request-local.
     */
    @Override
    public GenerationSession open(UUID sessionId, int[] promptTokens, GeneratorParameters parameters) {
        if (model.usesKvCache2Generation()) {
            return new LocalKvCache2GenerationSession(promptTokens, parameters);
        }
        return new LocalGenerationSession(promptTokens, parameters);
    }

    private final class LocalKvCache2GenerationSession implements GenerationSession {
        private final int[] promptTokens;
        private final KvCacheSession kvSession;

        private LocalKvCache2GenerationSession(int[] promptTokens, GeneratorParameters parameters) {
            this.promptTokens = promptTokens;
            this.kvSession = model.newKvCacheSession();
        }

        @Override
        public int prefixLength() {
            return 0;
        }

        @Override
        public AbstractTensor prefill(GenerationCursor cursor) {
            AbstractTensor last;
            if (cursor.hasTokensToProcess()) {
                last = model.batchForward(cursor.tokensToProcess(), cursor.startPosition(), kvSession);
            } else {
                kvSession.crop(cursor.replayPosition());
                last = model.forward(cursor.replayToken(), cursor.replayPosition(), kvSession);
            }
            model.emitGenerationDebug(new AbstractModel.GenerationDebugEvent(
                    AbstractModel.GenerationDebugEventType.AFTER_PROMPT_PREFILL,
                    promptTokens,
                    0,
                    cursor.startPosition(),
                    cursor.tokensToProcess().length,
                    null));
            return last;
        }

        @Override
        public AbstractTensor decode(int tokenId, int position) {
            return model.forward(tokenId, position, kvSession);
        }

        @Override
        public void afterDecode() {
        }

        @Override
        public void close() {
            kvSession.close();
        }
    }

    /** Per-request local KV state for {@link LocalGenerationBackend}. */
    private final class LocalGenerationSession implements GenerationSession {
        private final int[] promptTokens;
        private final KvBufferCache.KvBuffer kvBuffer;

        private LocalGenerationSession(int[] promptTokens, GeneratorParameters parameters) {
            this.promptTokens = promptTokens;
            this.kvBuffer = model.kvBufferCache.getEphemeralKvBuffer();
        }

        @Override
        public int prefixLength() {
            return 0;
        }

        /**
         * Runs local prompt prefill from the cursor start position.
         */
        @Override
        public AbstractTensor prefill(GenerationCursor cursor) {
            kvBuffer.setCurrentContextPosition(cursor.startPosition());
            AbstractTensor last;
            if (cursor.hasTokensToProcess()) {
                last = model.batchForward(cursor.tokensToProcess(), cursor.startPosition(), kvBuffer);
            } else {
                last = model.forward(cursor.replayToken(), cursor.replayPosition(), kvBuffer);
            }
            model.emitGenerationDebug(new AbstractModel.GenerationDebugEvent(
                    AbstractModel.GenerationDebugEventType.AFTER_PROMPT_PREFILL,
                    promptTokens,
                    0,
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
