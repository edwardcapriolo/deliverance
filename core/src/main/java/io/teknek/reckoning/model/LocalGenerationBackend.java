package io.teknek.reckoning.model;

import io.teknek.reckoning.generator.GeneratorParameters;
import io.teknek.reckoning.tensor.kv.KvCacheSession;
import io.teknek.reckoning.tensor2.TensorRef;

import java.util.Optional;
import java.util.UUID;

/**
 * Local generation backend backed by one in-process transformer executor.
 *
 * <p>This backend owns the local KV state for a generation request. The shared {@link GenerationEngineRef} handles token
 * sampling and stop conditions.</p>
 */
public final class LocalGenerationBackend implements GenerationBackendRef {
    private final AbstractModel model;

    public LocalGenerationBackend(AbstractModel model) {
        this.model = java.util.Objects.requireNonNull(model, "model");
    }

    @Override
    public void close() {
    }

    @Override
    public GenerationSessionRef openRef(UUID sessionId, int[] promptTokens, GeneratorParameters parameters) {
        if (!model.usesKvCache2Generation()) {
            throw new UnsupportedOperationException("TensorRef generation requires KV-cache v2");
        }
        return new LocalTensorRefGenerationSession(promptTokens, parameters);
    }

    private final class LocalTensorRefGenerationSession implements GenerationSessionRef {
        private final int[] promptTokens;
        private final Optional<String> effectiveCacheSalt;
        private final KvCacheSession kvSession;
        private final int prefixLength;

        private LocalTensorRefGenerationSession(int[] promptTokens, GeneratorParameters parameters) {
            this.promptTokens = promptTokens;
            this.effectiveCacheSalt = withActiveAdapterScope(parameters.cacheSalt);
            this.kvSession = model.newKvCacheSession();
            this.prefixLength = model.restoreSharedPrefixToKvSession(promptTokens, effectiveCacheSalt, kvSession);
            if (prefixLength > 0) {
                model.emitGenerationDebug(new AbstractModel.GenerationDebugEvent(
                        AbstractModel.GenerationDebugEventType.AFTER_PREFIX_COPY,
                        promptTokens, prefixLength, prefixLength, promptTokens.length - prefixLength));
            }
        }

        @Override
        public int prefixLength() {
            return prefixLength;
        }

        @Override
        public TensorRef prefill(GenerationCursor cursor) {
            TensorRef last;
            if (cursor.hasTokensToProcess()) {
                last = model.batchForwardRef(cursor.tokensToProcess(), cursor.startPosition(), kvSession);
                model.storeSharedPrefixFromKvSession(promptTokens, kvSession, effectiveCacheSalt);
            } else {
                kvSession.crop(cursor.replayPosition());
                last = model.forwardRef(cursor.replayToken(), cursor.replayPosition(), kvSession);
            }
            return last;
        }

        @Override
        public TensorRef decode(int tokenId, int position) {
            return model.forwardRef(tokenId, position, kvSession);
        }

        @Override
        public void close() {
            kvSession.close();
        }
    }

    /**
     * Namespaces shared KVCache2 prefix blocks by active LoRA adapter so cached KV state is not reused across different
     * effective weights.
     */
    private Optional<String> withActiveAdapterScope(Optional<String> callerSalt) {
        Optional<String> adapterId = model.activeLoraAdapterId();
        if (adapterId.isEmpty()) {
            return callerSalt;
        }
        return Optional.of("lora:" + adapterId.get() + "|" + callerSalt.orElse(""));
    }

}
