package io.teknek.deliverance.tensor.kv;

import io.teknek.deliverance.tensor.AbstractTensor;

/** Immutable logical read view over a KV cache session. */
public final class KvReadView implements AutoCloseable {
    private final KvCacheSession session;
    private final int layer;
    private final int visibleTokens;
    private final AttentionPattern pattern;

    KvReadView(KvCacheSession session, int layer, int visibleTokens, AttentionPattern pattern) {
        this.session = session;
        this.layer = layer;
        this.visibleTokens = visibleTokens;
        this.pattern = pattern;
    }

    public int layer() {
        return layer;
    }

    public int visibleTokens() {
        return visibleTokens;
    }

    public AttentionPattern pattern() {
        return pattern;
    }

    public AbstractTensor keyRowCopy(int position) {
        return session.keyRowCopy(layer, position);
    }

    public AbstractTensor valueRowCopy(int position) {
        return session.valueRowCopy(layer, position);
    }

    public AbstractTensor copyVisibleKeys() {
        return session.copyVisibleKeys(layer, visibleTokens);
    }

    public AbstractTensor copyVisibleValues() {
        return session.copyVisibleValues(layer, visibleTokens);
    }

    @Override
    public void close() {
    }
}
