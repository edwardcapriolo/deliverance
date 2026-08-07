package io.teknek.deliverance.tensor;

/**
 * Storage-side hook for KV cache rows.
 *
 * <p>The CPU backend is a no-op because {@link KvBufferCache} already writes CPU tensors directly. GPU-backed
 * implementations can mirror or own KV storage without leaking GPU lifecycle details into KV cache policy code.</p>
 */
public interface KvStorageBackend extends AutoCloseable {

    KvStorageBackend NONE = new KvStorageBackend() {
        @Override
        public void rowWritten(AbstractTensor keyPage, AbstractTensor valuePage, int rowInPage, int rowWidth) {
        }

        @Override
        public boolean supportsGpuAttention() {
            return false;
        }

        @Override
        public void close() {
        }
    };

    void rowWritten(AbstractTensor keyPage, AbstractTensor valuePage, int rowInPage, int rowWidth);

    default boolean supportsGpuAttention() {
        return false;
    }

    @Override
    default void close() {
    }
}
