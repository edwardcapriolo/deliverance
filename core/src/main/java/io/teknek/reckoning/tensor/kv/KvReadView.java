package io.teknek.reckoning.tensor.kv;

import io.teknek.reckoning.tensor2.TensorRef;

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

    public io.teknek.reckoning.DType keyDType() {
        return session.keyDType();
    }

    public io.teknek.reckoning.DType valueDType() {
        return session.valueDType();
    }

    public TensorRef keyRowCopyRef(int position) {
        return session.keyRowCopyRef(layer, position);
    }

    public TensorRef valueRowCopyRef(int position) {
        return session.valueRowCopyRef(layer, position);
    }

    /**
     * Returns a non-copying read-only key row view. In tracked mode, closing the returned tensor asserts that the
     * underlying KV row was not mutated while borrowed.
     */
    public TensorRef keyRowRef(int position) {
        return session.keyRowViewRef(layer, position);
    }

    /**
     * Returns a non-copying read-only value row view. In tracked mode, closing the returned tensor asserts that the
     * underlying KV row was not mutated while borrowed.
     */
    public TensorRef valueRowRef(int position) {
        return session.valueRowViewRef(layer, position);
    }

    public TensorRef copyVisibleKeysRef() {
        return session.copyVisibleKeysRef(layer, visibleTokens);
    }

    public TensorRef copyVisibleValuesRef() {
        return session.copyVisibleValuesRef(layer, visibleTokens);
    }

    public void copyKeyRows(int positionStart, int rowCount, TensorRef destination, int destinationRowStart) {
        session.copyKeyRows(layer, positionStart, rowCount, destination, destinationRowStart);
    }

    public void copyValueRows(int positionStart, int rowCount, TensorRef destination, int destinationRowStart) {
        session.copyValueRows(layer, positionStart, rowCount, destination, destinationRowStart);
    }

    public TensorRef[] keyPageRefs() {
        return session.keyPageRefs(layer, visibleTokens);
    }

    public TensorRef[] valuePageRefs() {
        return session.valuePageRefs(layer, visibleTokens);
    }

    @Override
    public void close() {
    }
}
