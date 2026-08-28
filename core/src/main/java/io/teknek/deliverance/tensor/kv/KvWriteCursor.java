package io.teknek.deliverance.tensor.kv;

import io.teknek.deliverance.tensor.AbstractTensor;

/** Explicit write cursor for KV cache updates. */
public final class KvWriteCursor implements AutoCloseable {
    private final KvCacheSession session;
    private final CacheExecutionMode mode;
    private boolean closed;

    KvWriteCursor(KvCacheSession session, CacheExecutionMode mode) {
        this.session = session;
        this.mode = mode;
    }

    public void write(int layer, int position, AbstractTensor key, AbstractTensor value) {
        if (closed) {
            throw new IllegalStateException("KV write cursor is closed");
        }
        session.write(mode, layer, position, key, value);
    }

    public void advanceLength(int newLength) {
        if (closed) {
            throw new IllegalStateException("KV write cursor is closed");
        }
        session.advanceLength(newLength);
    }

    @Override
    public void close() {
        closed = true;
    }
}
