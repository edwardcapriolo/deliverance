package io.teknek.deliverance.tensor;

import com.google.common.base.Preconditions;

import java.util.Objects;

/**
 * Reusable visible key/value page index for one layer.
 *
 * <p>{@link KvBufferCache.KvBuffer} owns the actual KV page tensors. This object is only the current view of which
 * context pages are visible for a layer up to {@code upperBound}. In decode, {@code visibleRows} changes every token,
 * but the page list changes only when generation crosses a page boundary. That makes this object a useful bridge toward
 * a vLLM-style page table without rebuilding page arrays for every token.</p>
 */
public final class KvPageTable implements AutoCloseable {
    private final int layerIndex;
    private final int pageRows;
    private final AbstractTensor[] keyPages;
    private final AbstractTensor[] valuePages;
    private int upperBound;
    private int visibleRows;

    public KvPageTable(int layerIndex, int upperBound, int pageRows, AbstractTensor[] keyPages,
            AbstractTensor[] valuePages) {
        this.layerIndex = layerIndex;
        this.pageRows = pageRows;
        this.keyPages = Objects.requireNonNull(keyPages, "keyPages");
        this.valuePages = Objects.requireNonNull(valuePages, "valuePages");
        Preconditions.checkArgument(keyPages.length == valuePages.length, "key/value page count mismatch");
        updateUpperBound(upperBound);
    }

    void updateUpperBound(int upperBound) {
        this.upperBound = upperBound;
        this.visibleRows = upperBound + 1;
    }

    public int layerIndex() {
        return layerIndex;
    }

    public int upperBound() {
        return upperBound;
    }

    public int visibleRows() {
        return visibleRows;
    }

    public int pageRows() {
        return pageRows;
    }

    public int pageCount() {
        return keyPages.length;
    }

    public AbstractTensor[] keyPages() {
        return keyPages;
    }

    public AbstractTensor[] valuePages() {
        return valuePages;
    }

    @Override
    public void close() {
        // The underlying page tensors are owned by KvBuffer. This is a reusable view, not an owner.
    }
}
