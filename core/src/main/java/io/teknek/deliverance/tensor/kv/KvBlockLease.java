package io.teknek.deliverance.tensor.kv;

import java.util.concurrent.atomic.AtomicBoolean;

/** Session-owned retained handle to a manager-owned immutable KV block. */
public final class KvBlockLease implements AutoCloseable {
    private final KvBlockManager.ManagedKvBlock managedBlock;
    private final String sessionId;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    KvBlockLease(KvBlockManager.ManagedKvBlock managedBlock, String sessionId) {
        this.managedBlock = managedBlock;
        this.sessionId = sessionId;
    }

    public KvBlockKey key() {
        return managedBlock.key();
    }

    public KvBlock block() {
        return managedBlock.block();
    }

    public String sessionId() {
        return sessionId;
    }

    public int blockIdentity() {
        return System.identityHashCode(block());
    }

    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            managedBlock.release();
        }
    }
}
