package io.teknek.deliverance.tensor.kv;

import io.teknek.deliverance.tensor.AbstractTensor;
import io.teknek.deliverance.tensor2.TensorRef;

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

    public io.teknek.deliverance.DType keyDType() {
        return session.keyDType();
    }

    public io.teknek.deliverance.DType valueDType() {
        return session.valueDType();
    }

    public AbstractTensor keyRowCopy(int position) {
        return session.keyRowCopy(layer, position);
    }

    public TensorRef keyRowCopyRef(int position) {
        return session.keyRowCopyRef(layer, position);
    }

    public AbstractTensor valueRowCopy(int position) {
        return session.valueRowCopy(layer, position);
    }

    public TensorRef valueRowCopyRef(int position) {
        return session.valueRowCopyRef(layer, position);
    }

    /**
     * Returns a non-copying read-only key row view. In tracked mode, closing the returned tensor asserts that the
     * underlying KV row was not mutated while borrowed.
     */
    public AbstractTensor keyRow(int position) {
        return session.keyRowView(layer, position);
    }

    public TensorRef keyRowRef(int position) {
        return TensorRef.owned(session.keyRowView(layer, position));
    }

    /**
     * Returns a non-copying read-only value row view. In tracked mode, closing the returned tensor asserts that the
     * underlying KV row was not mutated while borrowed.
     */
    public AbstractTensor valueRow(int position) {
        return session.valueRowView(layer, position);
    }

    public TensorRef valueRowRef(int position) {
        return TensorRef.owned(session.valueRowView(layer, position));
    }

    public AbstractTensor copyVisibleKeys() {
        return session.copyVisibleKeys(layer, visibleTokens);
    }

    public TensorRef copyVisibleKeysRef() {
        return TensorRef.owned(session.copyVisibleKeys(layer, visibleTokens));
    }

    public AbstractTensor copyVisibleValues() {
        return session.copyVisibleValues(layer, visibleTokens);
    }

    public TensorRef copyVisibleValuesRef() {
        return TensorRef.owned(session.copyVisibleValues(layer, visibleTokens));
    }

    public void copyKeyRows(int positionStart, int rowCount, AbstractTensor destination, int destinationRowStart) {
        session.copyKeyRows(layer, positionStart, rowCount, destination, destinationRowStart);
    }

    public void copyValueRows(int positionStart, int rowCount, AbstractTensor destination, int destinationRowStart) {
        session.copyValueRows(layer, positionStart, rowCount, destination, destinationRowStart);
    }

    public AbstractTensor[] keyPages() {
        return session.keyPages(layer, visibleTokens);
    }

    public TensorRef[] keyPageRefs() {
        return ownedRefs(session.keyPages(layer, visibleTokens));
    }

    public AbstractTensor[] valuePages() {
        return session.valuePages(layer, visibleTokens);
    }

    public TensorRef[] valuePageRefs() {
        return ownedRefs(session.valuePages(layer, visibleTokens));
    }

    private static TensorRef[] ownedRefs(AbstractTensor[] tensors) {
        TensorRef[] refs = new TensorRef[tensors.length];
        try {
            for (int i = 0; i < tensors.length; i++) {
                refs[i] = TensorRef.owned(tensors[i]);
            }
            return refs;
        } catch (RuntimeException | Error e) {
            for (TensorRef ref : refs) {
                if (ref != null) {
                    ref.close();
                }
            }
            throw e;
        }
    }

    @Override
    public void close() {
    }
}
