package io.teknek.deliverance.tensor;

/** Original references used a pair. This is more idiomatic. Even the generic type here is odd. Probably should be removed*/
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
