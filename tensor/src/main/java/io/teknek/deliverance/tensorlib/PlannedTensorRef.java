package io.teknek.deliverance.tensorlib;

import io.teknek.deliverance.tensor2.TensorRef;

/** A TensorRef materialization paired with its logical TensorPlan lineage. */
public record PlannedTensorRef(TensorRef tensor, TensorPlan.Tensor plan) implements AutoCloseable {
    @Override
    public void close() {
        tensor.close();
    }
}
