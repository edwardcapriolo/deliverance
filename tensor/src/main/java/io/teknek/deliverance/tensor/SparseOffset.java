package io.teknek.deliverance.tensor;

/**
 * Small pair type used by sparse tensor shapes.
 *
 * <p>Important: in the current tensor code this is usually interpreted as {@code (start, length)}, not
 * {@code (start, exclusiveEnd)}. The accessor name {@link #getEnd()} is therefore misleading for the main
 * tensor-shape use case, but is preserved because existing code already depends on it.</p>
 *
 * <p>When tensor docs say a row or column window is <em>resident</em>, they mean that window is physically
 * present in the current tensor's backing storage. The full logical tensor may be larger than the resident
 * window represented by this pair.</p>
 */
public class SparseOffset<T> {
    public static<X> SparseOffset<X> of (X start, X end){
        return new SparseOffset<>(start, end);
    }
    private final T start;
    private final T end;

    public SparseOffset(T start, T end){
        this.start = start;
        this.end = end;
    }

    public T getStart() {
        return start;
    }

    public T getEnd() {
        return end;
    }
}
