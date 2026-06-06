package io.teknek.deliverance.model.tensorparallel;

public record StaticTensorParallelContext(int rank, int size) implements TensorParallelContext {
    public StaticTensorParallelContext {
        if (size < 1) {
            throw new IllegalArgumentException("size must be >= 1");
        }
        if (rank < 0 || rank >= size) {
            throw new IllegalArgumentException("rank must be between 0 and size - 1");
        }
    }
}
