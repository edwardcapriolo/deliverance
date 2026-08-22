package io.teknek.deliverance.model.tensorparallel;

public record TensorParallelNodeCapacity(String nodeId, int slots) {
    public TensorParallelNodeCapacity {
        if (nodeId == null || nodeId.isBlank()) {
            throw new IllegalArgumentException("nodeId is required");
        }
        if (slots < 1) {
            throw new IllegalArgumentException("slots must be >= 1");
        }
    }
}
